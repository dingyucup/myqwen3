import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.qwen3_model import Qwen3Model

def main():
    Qwen3Model.merge_and_save_lora_model()

if __name__ == "__main__":
    main()