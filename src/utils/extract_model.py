import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *

import torch
from src.model.qwen3_model import  Qwen3Model

# 提取模型参数（去掉优化器的参数）
def extract_base_model(
        raw_model_file = MODELS_DIR/BEST_MODEL_NAME,
        output_model_file = BASE_MODEL_DIR/BASE_MODEL_NAME
):
    model = Qwen3Model.from_pretrained(raw_model_file)
    torch.save({
        "model_state_dict": model.state_dict(),
    }, output_model_file)


def extract_lora_model(
        raw_model_file = MODELS_DIR/BEST_MODEL_NAME,
        output_model_file = LORA_MODELS_DIR/LORA_MODEL_NAME
):
    model = Qwen3Model.from_lora_pretrained(raw_model_file)
    torch.save({
        "model_state_dict": model.state_dict(),
    }, output_model_file)


if __name__ == "__main__":
    # extract_base_model()
    extract_lora_model()
