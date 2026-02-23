import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn

from src.transformer.attention import GroupQueryAttention
from src.configuration.config import CONTEXT_LENGTH


class GroupQueryAttentionWithKVCache(nn.Module):
    def __init__(self, attention: GroupQueryAttention, context_length = CONTEXT_LENGTH):
        super().__init__()
        self.attn = attention
        self.context_length = context_length
        self.k_cache = None
        self.v_cache = None
        self.causal_mask = None
        self.position = 0 # 记录当前token的位置
        self._reset_kv_cache()

        self.register_buffer(
            "causal_mask_cache",
            torch.triu(
                torch.ones((context_length, context_length), dtype=torch.bool),
                diagonal=1
            ),
            persistent=False  # 不保存到state_dict
        )

    # 重置缓存
    def _reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.position = 0

    def reset_cache(self):
        self._reset_kv_cache()

    def _get_causal_mask(self, current_seq_len, total_seq_len):
        """
        获取因果掩码
        Args:
            current_seq_len: 当前处理的序列长度（通常是1）
            total_seq_len: 总序列长度（包括缓存）
        Returns:
            mask: [current_seq_len, total_seq_len] 的布尔掩码
        """
        # 从预分配的掩码中切片
        start_row = total_seq_len - current_seq_len
        mask = self.causal_mask_cache[start_row:total_seq_len, :total_seq_len]
        return mask

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # 如果有提示词处理提示词,处理提示词
        if self.k_cache is None:
            # 预分配缓存
            self.k_cache = torch.zeros(
                batch_size, self.attn.num_kv_groups, self.context_length, self.attn.head_dim,
                device=x.device, dtype=x.dtype
            )

            self.v_cache = torch.zeros(
                batch_size, self.attn.num_kv_groups, self.context_length, self.attn.head_dim,
                device=x.device, dtype=x.dtype
            )

            current_seq_len = seq_len
        else:
            # 只处理最后一个token
            x = x[:, -1:, :]
            current_seq_len = 1

        # 计算Q, K, V的值
        # 经过Query矩阵映射
        queries = self.attn.W_query(x)
        # 经过Key矩阵映射
        keys = self.attn.W_key(x)
        # 经过Value矩阵映射
        values = self.attn.W_value(x)

        # 重塑
        # 原来横向拼接的是(batch_size, seq_len, num_heads * head_dim)需要把按头拆出来
        queries = queries.view(batch_size, current_seq_len, self.attn.num_heads, self.attn.head_dim)
        # 原始(batch_size, seq_len, num_kv_groups * num_dim)
        keys = keys.view(batch_size, current_seq_len, self.attn.num_kv_groups, self.attn.head_dim)
        # 同上
        values = values.view(batch_size, current_seq_len, self.attn.num_kv_groups, self.attn.head_dim)

        # 保证最后两个维度是[..., seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1,2)

        # q和k向量归一化
        if self.attn.q_norm:
            queries = self.attn.q_norm(queries)

        if self.attn.k_norm:
            keys = self.attn.k_norm(keys)

        # 将位置编码应用于Queries和keys，position用于定位单个token
        queries = self.attn.rope(queries, start_pos= self.position)
        keys = self.attn.rope(keys, start_pos= self.position)

        # 将新计算出来的key和query存入缓存
        self.k_cache[:, :,self.position:self.position + current_seq_len] = keys
        self.v_cache[:, :,self.position:self.position + current_seq_len] = values

        # 根据现在的位置和序列长度确定序列总长度
        total_seq_len = self.position + current_seq_len

        # 因为要进行注意力评分计算，评分与相对位置有关系，所以要知道完整的token序列，
        keys = self.k_cache[:, :, :total_seq_len]
        values = self.v_cache[:, :, :total_seq_len]

        # 每组按照group_size大小中复制出与头双来对应的keys和values
        # (batch_size, num_kv_groups * group_size,seq_len,head_dim)
        keys = keys.repeat_interleave(self.attn.group_size, dim=1)
        values = values.repeat_interleave(self.attn.group_size, dim=1)

        # 注意力评分
        # queries: (batch_size, num_heads, seq_len, head_dim)
        # keys:(batch_size, num_heads, seq_len, head_dim)
        # attn_scores: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(-1, -2)
        # 带上掩码
        # 确保掩码在正确的设备上
        if self.causal_mask_cache.device != x.device:
            self.causal_mask_cache = self.causal_mask_cache.to(x.device)

        mask_bool = self._get_causal_mask(current_seq_len, total_seq_len)
        # [1, 1, 1, total_seq_len]
        mask_bool = mask_bool.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        # 注意力权重
        # attn_weights: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)

        # 注意力计算
        # values:(batch_size, num_kv_groups * group_size,seq_len,head_dim)
        # context_vec: (batch_size, num_kv_groups * group_size,seq_len,head_dim)
        context_vec = attn_weights @ values
        # context_vec: (batch_size, seq_len, num_kv_groups * group_size, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # context_vec: (batch_size, seq_len, num_kv_groups * group_size * head_dim)
        context_vec = context_vec.contiguous().view(batch_size, current_seq_len, self.attn.d_out)

        # 更新位置
        self.position += current_seq_len

        # 返回投影, 恢复输入形成(batch_size, seq_len, d_in)
        return self.attn.out_proj(context_vec)

def replace_attention_with_kv_cache(
        model: nn.Module
):
    for name, module in model.named_children():
        if isinstance(module, GroupQueryAttention):
            setattr(model, name, GroupQueryAttentionWithKVCache(module))
        else:
            replace_attention_with_kv_cache(module)

def add_kv_cache_to_model(model):
    replace_attention_with_kv_cache(model)
    return model

# 重置kv缓存
def reset_kv_cache_in_model(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, GroupQueryAttentionWithKVCache):
            module.reset_cache()
    return model




if __name__ == "__main__":

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device_t)

    torch.manual_seed(123)

    inputs = torch.tensor(
        [[[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]],  # step     (x^6)

         [[0.43, 0.15, 0.89],  # Your     (x^1)
          [0.55, 0.87, 0.66],  # journey  (x^2)
          [0.57, 0.85, 0.64],  # starts   (x^3)
          [0.22, 0.58, 0.33],  # with     (x^4)
          [0.77, 0.25, 0.10],  # one      (x^5)
          [0.05, 0.80, 0.55]]  # step     (x^6)
        ]
    )

    inputs = inputs.to(device_t)

    print(inputs.shape)
    # 测试自注意力类



    # 测试并行的多头分组自带掩码的注意力类
    print("=" * 20, "测试并行多头分组自带掩码的注意力类", "=" * 20)
    groups_head_attn_t = GroupQueryAttention(
        d_in=3,
        num_heads=8,
        num_kv_groups=4,
        head_dim=4,
        context_length=6
    ).to(device_t)

    print(groups_head_attn_t)
    print(groups_head_attn_t(inputs))

    # 测试并行的多头分组自带掩码的注意力类
    print("=" * 20, "测试并行多头分组自带掩码的KV Cache注意力类", "=" * 20)
    attn_kv_cache_t = GroupQueryAttentionWithKVCache(groups_head_attn_t).to(device_t)

    print(attn_kv_cache_t(inputs))


