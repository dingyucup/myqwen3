import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.configuration.config import *

class GPTDataset(Dataset):
    def __init__(self, data):
        self._data:dict = data

    @classmethod
    def from_json(
            cls,
            data_file = PREPROCESSED_DATA_DIR/TRAIN_DATA_FILE
    ):
        data_dict = pd.read_json(data_file, orient="records", lines=True).to_dict(orient="records")
        return cls(data_dict)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self._data[idx]["input"], dtype=torch.long)
        target_ids = torch.tensor(self._data[idx]["target"], dtype=torch.long)

        return input_ids, target_ids

def create_data_loader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0
):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader

def create_instruct_data_loader(
        dataset: Dataset,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        drop_last: bool = True,
        pad_token_id = None,
        max_seq_len = SEQ_LENGTH,
        num_workers: int = 0,
):
    assert pad_token_id is not None, "pad_token_id未指定"
    # 每个批次长度补全，同时做截断
    def custom_collate_fn(batch):
        inputs_list = []
        targets_list = []
        for item in batch:
            # # 最大填充
            # padding_inputs = torch.tensor([pad_token_id] * (max_seq_len - len(item[0])))
            # padding_targets = torch.tensor([pad_token_id] * (max_seq_len - len(item[1])))
            # inputs = torch.cat((item[0], padding_inputs), dim=-1)
            # targets = torch.cat((item[1], padding_targets), dim=-1)
            inputs = item[0] if len(item[0]) < max_seq_len else item[:max_seq_len]
            targets = item[1] if len(item[1]) < max_seq_len else item[:max_seq_len]
            inputs_list.append(inputs)
            targets_list.append(targets)

        inputs_tensor = pad_sequence(inputs_list, batch_first=True, padding_value=pad_token_id)
        targets_tensor = pad_sequence(targets_list, batch_first=True, padding_value=pad_token_id)

        # inputs_tensor = torch.stack(inputs_list, dim=0)
        # targets_tensor = torch.stack(targets_list, dim=0)
        return inputs_tensor, targets_tensor


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return data_loader

if __name__ == "__main__":
    from src.preprocess.tokenizer import Tokenizer
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    dataset_t = GPTDataset.from_json(data_file=str(INSTRUCT_PROCESSED_DIR/TRAIN_DATA_FILE))
    # print(dataset_t[0])

    data_loader_t = create_instruct_data_loader(
        dataset=dataset_t,
        pad_token_id=tokenizer_t.pad_id,
        shuffle=False
    )

    data_iter_t = iter(data_loader_t)
    inputs_t, targets_t = next(data_iter_t)
    print("Inputs:")
    print(inputs_t)
    print("Targets: ")
    print(targets_t)
    print(inputs_t.shape)
    print(targets_t.shape)
    # print(next(data_iter_t))
    # print(next(data_iter_t))
    # print(next(data_iter_t))