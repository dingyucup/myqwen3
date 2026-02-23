import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import torch
from torch import nn
from dataclasses import dataclass

from src.configuration.config import *
from src.preprocess.tokenizer import Tokenizer
from src.transformer.attention import GroupQueryAttention
from src.transformer.layer_norm import RMSNorm
from src.transformer.feed_forward import MoEFeedForward
from src.lora.lora import create_lora_model, merge_linear_with_lora


@dataclass
class Qwen3Config:
    vocab_size: int = 54800 # 词表大小
    context_length: int = CONTEXT_LENGTH # 上下文长度
    emb_dim: int = EMB_DIM # 词向量维度
    head_dim: int = HEAD_DIM # 头维度
    n_heads: int = N_HEADS # 注意力头的数量
    n_layers: int = N_LAYERS # 层数
    n_experts: int = N_EXPERTS # 专家数
    moe_intermediate_size: int = MOE_INTERMEDIATE_SIZE # 专家层的维度
    num_experts_per_token: int = NUM_EXPERTS_PER_TOKEN # 一个token需要几位专家
    num_kv_groups: int = N_KV_GROUPS # 把注意力头分成几组
    qk_norm: bool = QK_NORM

# 获取词表大小
vocab_size = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET).vocab_size
qwen3_config = Qwen3Config(vocab_size = vocab_size)


# Transformer块组件
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        # 带掩码的分组查询多头注意力子层
        self.attention = GroupQueryAttention(
            d_in=cfg.emb_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            num_kv_groups=cfg.num_kv_groups,
            qk_norm=cfg.qk_norm,
            context_length=cfg.context_length
        )

        # 前馈网络
        self.feedforward_network = MoEFeedForward(
            emb_dim=cfg.emb_dim,
            moe_intermediate_size=cfg.moe_intermediate_size,
            num_experts=cfg.n_experts,
            num_experts_per_token=cfg.num_experts_per_token
        )

        # 归一化层
        self.norm_1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm_2 = RMSNorm(cfg.emb_dim, eps=1e-6)


    def forward(self, x):
        shortcut = x
        # 前向传播, pre-norm->注意力子层
        x = self.norm_1(x)
        x = self.attention(x)
        x = x + shortcut # 残差连接（快捷连接）

        # 前向传播，前馈网络层
        shortcut = x
        x = self.norm_2(x)
        x = self.feedforward_network(x)
        # 残差连接
        x = x + shortcut

        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.config = cfg
        # 词嵌入层
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        # transformers层
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # 最终归一化层
        self.final_norm = RMSNorm(cfg.emb_dim)
        # 输出头, 恢复词表维度
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, inputs: torch.Tensor):
        x = self.token_embedding(inputs)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @classmethod
    def from_config(cls, qwen3_cfg: Qwen3Config= qwen3_config):
        return cls(qwen3_cfg)

    @classmethod
    def from_pretrained(cls, pretrained_model_file =MODELS_DIR / "best_model.pt"):
        model = cls(qwen3_config)
        state_dict = torch.load(pretrained_model_file)
        model.load_state_dict(state_dict["model_state_dict"])
        return model

    @classmethod
    def from_lora_pretrained(
            cls,
            pretrained_model_file =LORA_MODELS_DIR/LORA_MODEL_NAME,
            rank = RANK,
            alpha = ALPHA
    ):
        model = cls(qwen3_config)
        model = create_lora_model(model, rank=rank, alpha=alpha)
        state_dict = torch.load(pretrained_model_file)
        model.load_state_dict(state_dict["model_state_dict"])
        return model

    @classmethod
    def merge_and_save_lora_model(
            cls,
            model_with_lora=LORA_MODELS_DIR / LORA_MODEL_NAME,
            output_model=LORA_MODELS_DIR / MERGED_LORA_MODEL_NAME
    ):
        # 加载LoRA训练后的模型
        model = Qwen3Model.from_lora_pretrained(model_with_lora)

        before_merged_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = merge_linear_with_lora(model)
        torch.save({"model_state_dict": model.state_dict()}, output_model)
        print(model)
        # 恢复所有可训练参数
        for param in model.parameters():
            param.requires_grad = True

        after_merged_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"归并前模型可训练的参数量：{before_merged_params:,}")
        print(f"归并后模型可训练的参数量：{after_merged_params:,}")
        print(f"保存至：{output_model}")


if __name__ == "__main__":
    torch.manual_seed(42)
    # 分词器
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)

    # 模型参数设置
    config_t = Qwen3Config(vocab_size=tokenizer_t.vocab_size)
    model_t = Qwen3Model(config_t)


    # 计算模型参数数量
    print("Token 嵌入层形状:", model_t.token_embedding.weight.shape)
    print("Output 输出层形状:", model_t.out_head.weight.shape)

    total_params = sum(p.numel() for p in model_t.parameters())
    print(f"模型总参数量: {total_params:,}")

    total_params_qwen3 = (
            total_params - sum(p.numel()
                               for p in model_t.out_head.parameters())
    )
    print(f"可训练参数数量 "
          f"考虑权重共享: {total_params_qwen3}"
          )

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"模型总大小: {total_size_mb:.2f} MB")















