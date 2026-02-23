import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import math
import torch

import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, d_in, d_out, rank, alpha):
        super().__init__()
        # 构造A矩阵
        self.A = nn.Parameter(torch.empty(d_in, rank))
        # He初始化A矩阵
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # 构造全零的B矩阵
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        # 系数alpha
        self.alpha = alpha
        # 秩
        self.rank = rank

    def forward(self, x):
        scaling = self.alpha / math.sqrt(self.rank)
        return scaling * (x @ self.A @ self.B)



class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(
        model: nn.Module,
        rank: int,
        alpha: float,
        exclude_patterns = None,
        prefix = ""
):
    # 不给MoE加LoRA
    if exclude_patterns is None:
        exclude_patterns = ["feedforward_network"]


    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False


    # 线性层替换为LoRA层
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name  # 构造完整路径
        # 检查当前模块路径是否匹配任何排除模式
        should_exclude = any(pattern in full_name for pattern in exclude_patterns)

        if isinstance(module, nn.Linear):
            # 将需要训练层恢复梯度计算
            if should_exclude:
                # print(f"跳过:{full_name}")
                for param in module.parameters():
                    # 将梯度计算开回来
                    param.requires_grad = True
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归子层，替换掉线性层
            replace_linear_with_lora(module, rank, alpha, exclude_patterns, full_name)

    return model


def merge_linear_with_lora(model: nn.Module) -> nn.Module:
    """
    将模型中所有 LinearWithLoRA 模块合并为普通的 nn.Linear，
    并移除 LoRA 参数。
    """

    for name, module in list(model.named_children()):  # 使用 list 避免迭代时修改字典
        if isinstance(module, LinearWithLoRA):
            # 提取原始线性层和 LoRA 参数
            linear = module.linear
            lora = module.lora

            # 计算缩放因子
            scaling = lora.alpha / math.sqrt(lora.rank)

            # 计算LoRA贡献的权重增量（注意转置！）
            # A: [in_features, rank], B: [rank, out_features]
            # A @ B: [in_features, out_features]
            # 但 linear.weight 是 [out_features, in_features]
            delta_weight = (scaling * torch.mm(lora.A, lora.B)).T  # [out_features, in_features]

            # 创建新的Linear层
            merged_linear = nn.Linear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=linear.bias is not None,
                dtype=linear.weight.dtype,
                device=linear.weight.device
            )

            # 合并权重
            with torch.no_grad():
                merged_linear.weight.copy_(linear.weight + delta_weight)
                if linear.bias is not None:
                    merged_linear.bias.copy_(linear.bias)

            # 替换模块
            setattr(model, name, merged_linear)
        else:
            # 递归处理子模块
            merge_linear_with_lora(module)


    return model


def create_lora_model(model: nn.Module, rank: int, alpha: float):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"原始模型可训练的参数量：{total_params:,}")

    model = replace_linear_with_lora(model, rank, alpha)

    # 再次统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LoRA模型可训练的参数量：{total_params:,}")

    return model





if __name__ == "__main__":
    from src.model.qwen3_model import Qwen3Model
    from src.configuration.config import *
    base_model = Qwen3Model.from_pretrained(BASE_MODEL_DIR/BASE_MODEL_NAME)
    lora_model = create_lora_model(base_model, 16, 32)

