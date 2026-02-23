
import torch
from torch import nn
from src.configuration.config import *

# 假设输入的行向量是6维，系列长度为6，即6个token:我哥，喜欢，在家，洗脸，刷牙，唱歌
#   行向量乘以旋转矩阵:
#   我哥[x11, x12, x13, x14, x15, x16]  [  cosα   sinα    0      0      0      0    ]
#   喜欢[x21, x22, x23, x24, x25, x26]  [ -sinα   cosα    0      0      0      0    ]
#   在家[x31, x32, x33, x34, x35, x36]  [  0      0      cosβ   sinβ    0      0    ]
#   洗脸[x41, x42, x43, x44, x45, x46]  [  0      0      -sinβ  cosβ    0      0    ]
#   刷牙[x51, x52, x53, x54, x55, x56]  [  0      0       0      0      cosθ   sinθ ]
#   唱歌[x61, x62, x63, x64, x65, x66]  [  0      0       0      0      -sinθ  cosθ ]
# 我们把一个行向量中x00和x01配对，x02和x03配对……这样，就可以把(x00,x01)看作二维平面上的一个点，
# 这个点对应一个二维旋转矩阵 [ cosα   sinα] x1' = x1cosα - x2sinα, x2' = x1sinα + x2cosα
#                        [-sinα   cosα]


# 旋转位置编码
class RoPE(nn.Module):
    def __init__(
            self,
            head_dim,
            theta_base = 10000,
            context_length = CONTEXT_LENGTH,
    ):
        super().__init__()
        # 因为要对向量中的元素进行两两配对所以要求必须是偶数
        assert head_dim % 2 == 0, "向量维度必须是偶数"

        # 根据向量的维度数，计算每两个维度对应旋转角度
        # 第一步确向量元素x_2i对应的编号：1/(1000^(2i/d)), 因为两两配对，x_2i和x_{2i+1}共用一个
        vec_pos_encode = 1.0 / (theta_base**(torch.arange(0, head_dim, 2) / head_dim))
        # 第二步确定序列中token编号token_pos
        token_pos = torch.arange(context_length)
        # 第三步计算旋转角度pos/(1000^(2i/d)), 注意每个token有一个token_pos
        # 要不token_pos作用在token对应的向量元素的位置上，即x_pos_encode
        # 所以 angles形状为token个数 × (向量维度head_dim/2)
        angles = token_pos.unsqueeze(1) * vec_pos_encode.unsqueeze(0)

        # 计算正弦，余弦值， 并注册（相邻位置共用一个旋转角度）
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x, start_pos=0):
        batch_size, num_heads, seq_len, head_dim = x.shape
        # 检查头的维度数
        assert head_dim % 2 == 0, "头的维度数不是偶数"
        assert start_pos + seq_len <= self.cos.shape[0], \
            f"长度{seq_len}从位置{start_pos}开始计算超过最大长度"

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        assert seq_len <= self.cos.shape[0], \
            f"序列长度 {seq_len} 超过预设的 context_length {self.cos.shape[0]}"

        # 调整形状：1, 1, seq_len, head_dim
        cos = self.cos[start_pos:start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start_pos:start_pos + seq_len, :].unsqueeze(0).unsqueeze(0)

        # 旋转
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos


        # 整合
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x1_rot
        x_rot[..., 1::2] = x2_rot

        return x_rot



if __name__ == "__main__":
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device_t)
    rope_t =RoPE(8)
    rope_t.to(device_t)
    inputs_t = torch.tensor([[[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                                 ]]])
    inputs_t = inputs_t.to(device_t)
    print(rope_t(inputs_t))

