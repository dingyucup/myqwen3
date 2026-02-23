import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

from src.configuration.config import *
from src.preprocess.tokenizer import Tokenizer
import pandas as pd

def format_preference_input(entry):
    instruction_text = (
        f"### 指令：\n你是一个知识渊博的助手，请基于事实，准确、简洁地回答以下问题。"
    )

    input_text = (
        f"\n\n### 问题：\n{entry['prompt'] if entry['prompt'] else "" }\n\n### 答案：\n"
    )

    chosen_response_text = (
        f"{entry['chosen'] if entry['chosen'] else "" }"
    )

    rejected_response_text = (
        f"{entry['rejected'] if entry['rejected'] else "" }"
    )

    prompt_text = instruction_text + input_text
    # chosen_full_text = instruction_text + input_text + chosen_response_text
    # rejected_full_text = instruction_text + input_text + rejected_response_text

    return prompt_text, chosen_response_text, rejected_response_text


class PreferenceProcessor:
    def __init__(self, tokenizer: Tokenizer, max_trunc_len = 512):
        self._tokenizer = tokenizer
        self.max_trunc_len = max_trunc_len
    def encode_dataset(self, df_data):
        data = []
        for item in df_data:
            prompt_text, chosen_response_text, rejected_response_text = format_preference_input(item)
            # 提示词编码
            prompt_token_ids  = self._tokenizer.encode(prompt_text)
            # 答案编码
            chosen_token_ids = self._tokenizer.encode(chosen_response_text)
            rejected_token_ids = self._tokenizer.encode(rejected_response_text)

            # 答案编码，并增加结束标记
            chosen_full_token_ids = prompt_token_ids + chosen_token_ids + [self._tokenizer.eos_id]
            rejected_full_token_ids = prompt_token_ids + rejected_token_ids + [self._tokenizer.eos_id]
            # 截断
            # chosen_full_token_ids = chosen_full_token_ids \
            #     if len(chosen_full_token_ids) <= self.max_trunc_len \
            #     else chosen_full_token_ids[:self.max_trunc_len]
            #
            # rejected_full_token_ids = rejected_full_token_ids \
            #     if len(rejected_full_token_ids) <= self.max_trunc_len \
            #     else rejected_full_token_ids[:self.max_trunc_len]
            # 去掉太长的回复
            if len(chosen_full_token_ids) > self.max_trunc_len:
                continue
            elif len(rejected_full_token_ids) > self.max_trunc_len:
                continue

            # 把回答之前的内容掩码掉
            chosen_mask = ([False]*(len(prompt_token_ids))
                           + [True] * (len(chosen_full_token_ids) - len(prompt_token_ids)))
            rejected_mask = ([False]*(len(prompt_token_ids))
                           + [True] * (len(rejected_full_token_ids) - len(prompt_token_ids)))

            data.append(
                {
                    "chosen": chosen_full_token_ids,
                    "rejected": rejected_full_token_ids,
                    "chosen_mask": chosen_mask,
                    "rejected_mask": rejected_mask
                }
            )

        return data


    def processing(
            self,
            raw_data_file = DPO_RAW_DIR/DPO_PREFERENCE_RAW_DATA_FILE,
            output_dir = DPO_PROCESSED_DIR
    ):
        df = pd.read_json(
            str(raw_data_file),
            orient="records",
            lines=True
        )

        # 去除
        df = df.dropna(how="any")
        # lengths = df["prompt"].str.len()
        # max_length = lengths.quantile(0.95)
        # print(max_length)
        # return
        df = df.to_dict(orient="records")
        df_len = len(df)



        # 多线程处理
        groups = os.cpu_count() - 8 # 刨去8个超线程
        stride = df_len // groups
        if df_len % groups != 0:
            groups += 1

        data = []
        with multiprocessing.Pool(groups) as pool:
            results = [
                pool.apply_async(self.encode_dataset, kwds={"df_data": df[i:i + stride]})
                for i in range(0, df_len, stride)
            ]
            for result in tqdm(results, desc="处理数据块"):
                data.extend(result.get())


        # 划分训练集、验证集和测试集
        train_data, test_data = train_test_split(data, train_size=0.85, random_state=123)
        valid_data, test_data = train_test_split(test_data, test_size=0.33333,random_state=123)

        # 保存数据集
        pd.DataFrame(train_data).to_json(output_dir/"train.jsonl", orient="records", lines=True)
        pd.DataFrame(valid_data).to_json(output_dir/"valid.jsonl", orient="records", lines=True)
        pd.DataFrame(test_data).to_json(output_dir/"test.jsonl", orient="records", lines=True)
        print(f"数据集目录：{output_dir}")
        print(f"训练数据集大小：{len(train_data)}")
        print(f"验证数据集大小：{len(valid_data)}")
        print(f"测试数据集大小：{len(test_data)}")


if __name__ == "__main__":
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    prefer_processor = PreferenceProcessor(tokenizer=tokenizer_t, max_trunc_len=128)
    prefer_processor.processing()
