import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn

from src.transformer.rope import RoPE
from src.transformer.layer_norm import RMSNorm

#高效多头注意力类，为了能够多头并行进行改进
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            context_length,
            dropout,
            num_heads,
            qkv_bias = False
    ):
        super().__init__()
        # 将多头Q，K，V矩阵恒祥拼接成一个矩阵所以要求列数能被头数整除
        assert (d_out % num_heads == 0), "d_out不能被n_heads整除"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 单头向量维度
        # 自注意力层Q, K, V矩阵
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 将多头输出按某种权重输出
        self.out_proj = nn.Linear(d_out, d_out)
        # 随机失活
        self.dropout = nn.Dropout(dropout)

        # 自注意力掩码层,设置上三角矩阵，为方便矩阵转移到显卡上采用父类的register_buffer方法
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )



    def forward(self, x):
        # 记录输入数据的形状, 分别为批大小，文本序列长度，词向量维度
        batch_size, seq_len, emb_dim = x.shape
        # 将三个矩阵作用在输入x上，同时得到多头结果
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 为了能够进行点击运算，需要对矩阵进行重塑
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 交换维度，因为点击运算是每个token对应的Q,K,W中的向量之间的运算
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # 按照论文的公式
        # attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d_{model}}})V
        # d_model取最后一个维度，也就是一个词向量经过key矩阵作用后得到一个新的向量
        # 计算注意力评分
        attn_scores = queries @ keys.transpose(-1, -2)
        # 将矩阵右上角置为-inf, 可以在softmax计算中exp(-inf)=0，不必为每一行再做归一化
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf) # 稳妥起见
        # attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 注意力权重计算
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 随机失活, 同时按失活比例的倒数放大权重参数
        attn_weights = self.dropout(attn_weights)
        # 加权汇总输出, 形状要从（batch_size, self.num_heads, seq_len, vec_dim）
        # 变为 （batch_size, seq_len, self.num_heads, vec_dim）为了方便多头叠加
        context_vec = (attn_weights @ values).transpose(1,2)
        # 为了多头矩阵运算结果能够叠加，需要转为d_out的形状，转化前强制内存连续
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_out)

        # 对多头的输出，按某种权重输出
        result = self.out_proj(context_vec)

        return result


# 多头分组查询注意力类
class GroupQueryAttention(nn.Module):
    def __init__(
            self,
            d_in: int,
            num_heads: int,
            num_kv_groups: int,
            context_length: int,
            head_dim: int = None,
            qk_norm = True
    ):
        super().__init__()
        # 确保头数能被完美分组，所以要求头数能被组数整除
        assert num_heads % num_kv_groups == 0, "头数num_heads不能被组数num_kv_groups整除"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # 头数和组数都有了，就知道每组有几个头了
        self.group_size = num_heads // num_kv_groups

        # 如果没有设置头的维度数，我们需要计算一下头的维度数, 那就认为d_in和d_out相同
        if head_dim is None:
            # d_in的维度取决与几个头，为了多头并行，将多个头横着拼接成一个矩阵
            assert d_in % num_heads == 0, "因为你没有设置头的维度数head_dim,所以要求d_in能被头数整除"
            # 求得头的维度数
            head_dim = d_in // num_heads

        # 不管设置没设置，这时候我们都知道了头的维度数
        self.head_dim = head_dim
        # 计算一下输出维度
        self.d_out = num_heads * head_dim
        # 查询矩阵
        self.W_query = nn.Linear(d_in, self.d_out, bias=False)
        # 键矩阵, 因为同一个组中头共用一个键矩阵，所以输出维度就是组数乘以一个头的维度
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        # 值矩阵同理, 因为同一个组中头共用一个值矩阵，所以输出维度就是组数乘以一个头的维度
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False)
        # 旋转位置编码
        self.rope = RoPE(self.head_dim)
        # 做投影, 恢复原来的输入维度
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False)
        # 掩码
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # 计算Q, K, V的值
        # 经过Query矩阵映射
        queries = self.W_query(x)
        # 经过Key矩阵映射
        keys = self.W_key(x)
        # 经过Value矩阵映射
        values = self.W_value(x)

        # 重塑
        # 原来横向拼接的是(batch_size, seq_len, num_heads * head_dim)需要把按头拆出来
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 原始(batch_size, seq_len, num_kv_groups * num_dim)
        keys = keys.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        # 同上
        values = values.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)

        # 保证最后两个维度是[..., seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1,2)

        # q和k向量归一化
        if self.q_norm:
            queries = self.q_norm(queries)

        if self.k_norm:
            keys = self.k_norm(keys)

        # 将位置编码应用于Queries和keys
        queries = self.rope(queries)
        keys = self.rope(keys)

        # 每组按照group_size大小中复制出与头双来对应的keys和values
        # (batch_size, num_kv_groups * group_size,seq_len,head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 注意力评分
        # queries: (batch_size, num_heads, seq_len, head_dim)
        # keys:(batch_size, num_heads, seq_len, head_dim)
        # attn_scores: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = queries @ keys.transpose(-1, -2)
        # 带上掩码
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
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
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_out)

        # 返回投影, 恢复输入形成(batch_size, seq_len, d_in)
        return self.out_proj(context_vec)





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


    # 测试并行的多头自带掩码的注意力类
    print("=" * 20, "测试并行多头自带掩码的注意力类", "=" * 20)
    multi_head_attn_t = MultiHeadAttention(
        d_in=3,
        d_out=4,
        context_length=6,
        dropout=0.5,
        num_heads=2,
    )
    multi_head_attn_t.to(device_t)
    print(multi_head_attn_t)
    print(multi_head_attn_t(inputs))

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














