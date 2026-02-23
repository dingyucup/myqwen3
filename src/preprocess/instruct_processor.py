import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

from src.configuration.config import *
from src.preprocess.tokenizer import Tokenizer
import pandas as pd

def format_input(entry):
    instruction_text = (
        f"### 指令：\n你是一个知识渊博的助手，请基于事实，准确、简洁地回答以下问题。"
    )

    input_text = (
        f"\n\n### 问题：\n{entry["question"] if entry["question"] else "" }"
    )

    response_text = (
        f"\n\n### 答案：\n{entry["answer"] if entry["answer"] else "" }"
    )

    return  instruction_text + input_text + response_text


class InstructProcessor:
    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def encode_instruction(self, df_data, max_length):
        data_collection = []
        for entry in tqdm(df_data, desc="指令编码"):
            format_text = format_input(entry)
            format_text_ids = self._tokenizer.encode(format_text)
            # 构造 inputs: 截断到 max_length
            inputs = format_text_ids[:max_length]
            # 构造 targets: 右移一位 + eos
            targets = format_text_ids[1:max_length]
            targets.append(self._tokenizer.eos_id)
            data_collection.append({"input": inputs, "target": targets})

        return data_collection

    def processing(
            self,
            max_length=SEQ_LENGTH,
            input_txt_file=INSTRUCT_RAW_DIR/INSTRUCT_RAW_FILE,
            output_dir=INSTRUCT_PROCESSED_DIR,
    ):

        print("=" * 20, "数据处理开始", "=" * 20)
        # 加载数据集
        df = pd.read_json(
            str(input_txt_file),
            lines=True,
            orient="records",
            encoding="utf-8"
        )
        # 删除空列
        df = df.drop(columns=["history", "response"])
        # 删除数值为NAN
        df = df.dropna(how="any")
        # 转为字典
        df = df.to_dict(orient="records")
        # 计算数据集长度
        data_len = len(df)

        # 多进程处理
        groups = os.cpu_count()
        stride = data_len // groups
        if data_len % groups != 0:
            groups += 1

        data = []
        with multiprocessing.Pool(groups) as pool:
            results = [
                pool.apply_async(self.encode_instruction, kwds={"df_data": df[i:i + stride], "max_length": max_length})
                for i in range(0, data_len, stride)
            ]
            for result in tqdm(results, desc="处理数据块"):
                data.extend(result.get())

        # 划分数据集
        train_set, valid_set = train_test_split(
            data,
            train_size=TRAIN_SET_RATIO,
            random_state=123
        )

        # 保存数据集
        pd.DataFrame(train_set).to_json(output_dir/TRAIN_DATA_FILE, orient="records", lines=True)
        pd.DataFrame(valid_set).to_json(output_dir/VALID_DATA_FILE, orient="records", lines=True)
        print(f"数据集目录：{output_dir}")
        print(f"训练数据集大小：{len(train_set)}")
        print(f"验证数据集大小：{len(valid_set)}")

        print("=" * 20, "数据处理结束", "=" * 20)


if __name__ == "__main__":
    from src.preprocess.tokenizer import Tokenizer

    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    processor = InstructProcessor(tokenizer_t)
    processor.processing()
