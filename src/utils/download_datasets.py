import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *
from datasets import load_dataset

def download_dpo_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset["train"].to_json(RAW_DATA_DIR/(dataset_name + ".jsonl"))



if __name__ == "__main__":
    download_dpo_dataset("m-a-p/COIG-P")
