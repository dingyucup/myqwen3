import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # 防零除
        # 缩放与平移
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        # 求均值
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 有偏估计
        # 归一化(x-μ)/σ
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # 除以标准差
        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
   def __init__(
           self,
           emb_dim,
           eps = 1e-6,
   ):
       """
       RMSNorm（Root Mean Square Normalization，均方根归一化） 是对传统 LayerNorm 的一种简化变体。
       它去除了均值标准化操作，仅基于输入特征的均方根（RMS）进行缩放，
       从而在保持归一化效果的同时降低计算复杂度。
       :param emb_dim: 向量维度
       :param eps: 防零除
       """
       super().__init__()
       self.eps = eps
        # 可学习的缩放矩阵
       self.scale = nn.Parameter(torch.ones(emb_dim))


   def forward(self, x: torch.Tensor):
       # 保存数据类型
       input_dtype = x.dtype
       x = x.to(torch.float32)
       # 根据均方根公式对每一行向量作归一化
       # 先求向量元素的平方和的平均值
       var = x.pow(2).mean(dim=-1, keepdim=True)
       # 之后向量中所有元素求乘以这个向量均方根的倒数，调用torch的rsqrt方法即可
       norm_x = x * torch.rsqrt(var + self.eps)
       norm_x = norm_x * self.scale

       return norm_x.to(input_dtype)



if __name__ == "__main__":
    from src.preprocess.tokenizer import Tokenizer
    from src.configuration.config import *
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    text_1 = "我深爱着你"
    text_2 = "我深爱着她"
    batch = [tokenizer_t.encode(text_1), tokenizer_t.encode(text_2)]
    x_batch = torch.tensor(batch)
    print(x_batch)
    print(x_batch.shape)
    embed_t = nn.Embedding(360, 1024)
    linear_t = nn.Linear(1024, 1024)
    print(embed_t.weight.shape)
    x_vec_batch = embed_t(x_batch)
    x_vec_batch = linear_t(x_vec_batch)
    print("="*20, "词向量", "="*20)
    print(x_vec_batch)
    # 均值和方差检查
    mean_t = x_vec_batch.mean(dim=-1, keepdim=True)
    var_t = x_vec_batch.var(dim=-1, keepdim=True, unbiased=False)
    print("归一化前的均值:\n", mean_t)
    print("归一化前的方差:\n", var_t)

    print("=" * 20, "测试均方根层归一化", "=" * 20)
    rms_norm_t = RMSNorm(1024)

    x_norm_t = rms_norm_t(x_vec_batch)

    # 经过归一化层后的输出
    print("归一化层后的输出:\n", x_norm_t)
    mean_layer = x_norm_t.mean(dim=-1, keepdim=True)
    var_layer = x_norm_t.var(dim=-1, keepdim=True, unbiased=False)
    print("归一化层后的均值:\n", mean_layer)
    print("归一化层后的方差:\n", var_layer)




