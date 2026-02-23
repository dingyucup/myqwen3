import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.configuration.config import *

class DPODataset(Dataset):
    def __init__(self, data):
        self._data:dict = data

    @classmethod
    def from_json(
            cls,
            data_file
    ):
        data_dict = pd.read_json(data_file, orient="records", lines=True).to_dict(orient="records")
        return cls(data_dict)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        chosen_ids = torch.tensor(self._data[idx]["chosen"], dtype=torch.long)
        rejected_ids = torch.tensor(self._data[idx]["rejected"], dtype=torch.long)
        chosen_mask = torch.tensor(self._data[idx]["chosen_mask"], dtype=torch.bool)
        rejected_mask = torch.tensor(self._data[idx]["rejected_mask"], dtype=torch.bool)

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask


def create_dpo_data_loader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = True,
        pad_token_id = None,
        num_workers: int = 0,
):
    assert pad_token_id is not None, "pad_token_id未指定"
    # 每个批次长度补全，同时做截断
    def custom_collate_fn(batch):
        chosen_list = []
        chosen_mask_list = []
        rejected_list = []
        rejected_mask_list = []
        for item in batch:
            chosen_list.append(item[0])
            chosen_mask_list.append(item[1])
            rejected_list.append(item[2])
            rejected_mask_list.append(item[3])

        chosen_tensor = pad_sequence(chosen_list, batch_first=True, padding_value=pad_token_id)
        chosen_mask_tensor = pad_sequence(chosen_mask_list, batch_first=True, padding_value=False)
        rejected_tensor = pad_sequence(rejected_list, batch_first=True, padding_value=pad_token_id)
        rejected_mask_tensor = pad_sequence(rejected_mask_list, batch_first=True, padding_value=False)


        return chosen_tensor, chosen_mask_tensor, rejected_tensor, rejected_mask_tensor


    dpo_data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return dpo_data_loader

if __name__ == "__main__":
    from src.preprocess.tokenizer import Tokenizer
    from src.model.qwen3_model import Qwen3Model
    import torch.nn.functional as F

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device_t}")
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    dataset_t = DPODataset.from_json(data_file=str(DPO_PROCESSED_DIR/"train.jsonl"))
    # 加载预训练后的LoRA模型
    policy_model_t = Qwen3Model.from_lora_pretrained()
    refer_model_t = Qwen3Model.from_lora_pretrained()
    policy_model_t = policy_model_t.to(device_t)
    # refer_model_t = refer_model_t.to(device_t).eval()

    data_loader_t = create_dpo_data_loader(
        dataset=dataset_t,
        pad_token_id=tokenizer_t.eos_id,
        shuffle=False
    )

    data_iter_t = iter(data_loader_t)
    chosen_t, chosen_mask_t, rejected_t, rejected_mask_t = next(data_iter_t)
    chosen_t = chosen_t.to(device_t)
    chosen_mask_t = chosen_mask_t.to(device_t)
    rejected_t = rejected_t.to(device_t)
    refer_model_t = refer_model_t.to(device_t)
    # print("chosen:")
    # print(chosen_t)
    # print("chosen_mask: ")
    # print(chosen_mask_t)
    # print(chosen_t.shape)
    # print(chosen_mask_t.shape)
    # print("========================")
    # print("rejected:")
    # print(rejected_t)
    # print("rejected_mask: ")
    # print(rejected_mask_t)
    # print(rejected_t.shape)
    # print(rejected_mask_t.shape)
    print("===========前向传播=============")
    chosen_labels = chosen_t[:, 1:].clone().unsqueeze(-1)
    chosen_logits = policy_model_t(chosen_t)
    chosen_logits = chosen_logits[:, :-1]
    log_probs = F.log_softmax(chosen_logits, dim=-1)
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=chosen_labels
    ).squeeze(-1)

    mask = chosen_mask_t[:, 1:].clone()

    selected_log_probs = selected_log_probs * mask

    avg_log_probs = selected_log_probs.sum(dim=-1) / mask.sum(dim=-1)

    print(avg_log_probs)
    print(avg_log_probs.shape)



    # print(chosen_labels.shape)
    # print(chosen_logits.shape)
    # print(chosen_mask_t.shape)
    # print(log_probs.shape)



