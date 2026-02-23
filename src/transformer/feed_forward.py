import  torch
from torch import nn

class MoEFeedForward(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            moe_intermediate_size: int,
            num_experts: int,
            num_experts_per_token: int,

    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.emb_dim = emb_dim
        # 门控激活函数, 也是专家路由
        # 输入: [batch_size, seq_len, emb_dim] -> 输出: [batch_size, seq_len, num_experts]
        self.gate = nn.Linear(emb_dim, num_experts, bias=False)

        # 专家网络组件:
        # fc1: 第一个线性层，用于SwiGLU门控机制的第一个分支
        self.fc1 = nn.ModuleList([
            nn.Linear(emb_dim, moe_intermediate_size, bias=False) for _ in range(num_experts)
        ])
        # fc2: 第二个线性层，用于SwiGLU门控机制的第二个分支
        self.fc2 = nn.ModuleList([
            nn.Linear(emb_dim, moe_intermediate_size, bias=False) for _ in range(num_experts)
        ])
        # fc3: 第三个线性层，将中间表示映射回原始维度
        self.fc3 = nn.ModuleList([
            nn.Linear(moe_intermediate_size, emb_dim, bias=False) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch_size, seq_len, num_experts)
        # 步骤1: 计算门控分数
        # scores: [batch_size, seq_len, num_experts] - 每个token对每个专家的原始分数
        scores = self.gate(x)

        # 步骤2: 为每个token选择top-k个专家
        # topk_scores: [batch_size, seq_len, num_experts_per_token] - top-k个专家的原始分数
        # topk_indices: [batch_size, seq_len, num_experts_per_token] - 对应的专家索引
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_token, dim=-1)

        # 步骤3: 将原始分数转换为概率分布(归一化)
        # topk_probs: [batch_size, seq_len, num_experts_per_tok] - 归一化后的专家权重
        topk_probs = torch.softmax(topk_scores, dim=-1)

        # 步骤4: 获取输入形状并展平，便于按token处理
        batch_size, seq_len, emb_dim = x.shape
        # x_flat: (batch_size * seq_len, emb_dim) - 展平后输入
        x_flat: torch.Tensor = x.reshape(batch_size * seq_len, -1)
        # out_flat :(batch_size*seq_len, emb_dim) - 初始化输出
        out_flat = torch.zeros((batch_size*seq_len, emb_dim), device=x.device)

        # 步骤5: 展平专家索引和权重，便于按token处理
        # topk_indices_flat: [batch * seq_len, num_experts_per_token]
        topk_indices_flat = topk_indices.reshape((-1, self.num_experts_per_token))
        # topk_probs_flat: [batch * seq_len, num_experts_per_token]
        topk_probs_flat = topk_probs.reshape((-1, self.num_experts_per_token))

        # 步骤6: 找出所有被选中的专家(去重)
        # unique_experts: 包含所有被至少一个token选中的专家ID
        unique_experts = torch.unique(topk_indices_flat)


        # 步骤7: 对每个被选中的专家进行处理
        for expert_id_tensor in unique_experts:
            # 专家编号转化为标量
            expert_id = expert_id_tensor.item()

            # 步骤8: 创建掩码，找出哪些token选择了当前专家
            # mask: [batch * seq_len, num_experts_per_tok] - 布尔张量
            mask = (topk_indices_flat == expert_id)

            # 如果没有token选择当前专家，跳过，看下一位专家
            if not mask.any():
                continue

            # 步骤9: 找出至少在一个位置选择了当前专家的token
            # token_mask: [batch * seq_len] - 布尔张量，标记哪些token选择了当前专家
            token_mask = mask.any(dim=-1)
            # selected_idx: 包含选择了当前专家的token索引, nonzero()：获取张量中非零（或True）元素的索引位置
            selected_idx = token_mask.nonzero(as_tuple = False).squeeze(-1)


            # 如果没有token选择当前专家(双重检查)，跳过, numel()：获取张量中元素的总数，常用于检查张量是否为空
            if selected_idx.numel() == 0:
                continue

            # 步骤10: 获取选择了当前专家的token的输入
            # expert_input: [num_selected_tokens, emb_dim]
            expert_input = x_flat.index_select(0, selected_idx)


            # 步骤11: 应用当前专家的前馈网络(SwiGLU激活)
            # 计算SwiGLU: SiLU(fc1(x)) * fc2(x)
            # hidden: [num_selected_tokens, moe_intermediate_size]
            fc1 = self.fc1[expert_id](expert_input)
            fc2 = self.fc2[expert_id](expert_input)
            hidden = torch.nn.functional.silu(fc1) * fc2
            # expert_out: [num_selected_tokens, emb_dim] - 专家输出
            expert_out = self.fc3[expert_id](hidden)

            # 步骤12: 获取每个token在当前专家的权重
            # mask_selected: [num_selected_tokens, num_experts_per_token] - 所选token的掩码
            mask_selected = mask[selected_idx]
            # slot_indices: [num_selected_tokens, 1] - 当前专家在每个token的top-k中的位置
            slot_indices = mask_selected.int().argmax(dim=-1, keepdims=True)
            # selected_probs: [num_selected_tokens] - 收集当前专家在每个token的权重
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx), # 在token方向上选出token
                dim=-1,
                index=slot_indices
            )
            # 步骤13: 加权累加专家输出到最终输出
            # out_flat: [batch * seq_len, emb_dim]
            # selected_idx: [num_selected_tokens] - 需要更新的token索引
            # expert_out: [num_selected_tokens, emb_dim] - 专家输出
            # selected_probs: [num_selected_tokens] - 权重
            out_flat = out_flat.index_add(0, selected_idx, expert_out * selected_probs)

        # 步骤14: 恢复原始形状并返回
        # out_flat: [batch * seq_len, emb_dim] -> [batch, seq_len, emb_dim]
        return out_flat.reshape(batch_size, seq_len, self.emb_dim)


if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.tensor(
        [[[0.2274, 1.0903, 2.1292, 0.2274, 0.8003, 3.0707, 0.2274, 0.2274],
          [0.3888, 0.3888, 0.3888, 0.3888, 0.3888, 0.9863, 1.6923, 3.3774],
          [2.8116, 0.4097, 0.4130, 2.6485, 0.4882, 0.4097, 0.4097, 0.4097],
          [0.8283, 0.4061, 0.4061, 1.7739, 0.4061, 0.4061, 0.4061, 3.3675]],

         [[0.2274, 1.0903, 2.1292, 0.2274, 0.8003, 3.0707, 0.2274, 0.2274],
          [0.2302, 1.8124, 0.9270, 0.2302, 0.2302, 3.2251, 0.2302, 1.1147],
          [0.6221, 0.6221, 0.6221, 3.6456, 0.6221, 0.6221, 0.6221, 0.6221],
          [0.4425, 0.4425, 0.4425, 0.4425, 0.4425, 0.4425, 2.1524, 3.1924]]]
    ).to(device)

    ffn_t = MoEFeedForward(
        emb_dim=8,
        moe_intermediate_size=6,
        num_experts=10,
        num_experts_per_token=2
    ).to(device)


    res = ffn_t(inputs)
    print(res)


