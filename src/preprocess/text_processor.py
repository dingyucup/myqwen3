import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.configuration.config import *
from src.preprocess.tokenizer import Tokenizer

import pandas as pd
class TextProcessor:
    def __init__(self, tokenizer:Tokenizer):
        self._tokenizer = tokenizer

    def processing(
            self,
            max_length = SEQ_LENGTH,
            stride = STRIDE,
            input_txt_dir = RAW_DATA_DIR/NOVELS_TXT_DIR,
            output_dir = PREPROCESSED_DATA_DIR,
    ):
        print("="*20, "数据处理开始", "="*20)


        # 罗列所有小说
        txt_files = [novel.name for novel in Path(input_txt_dir).glob("*.txt")]

        # 从小说中提取分词
        def text_encode(txt_name):
            txt = str(Path(input_txt_dir) / txt_name)

            try:
                with open(txt, "r", encoding="utf-8") as f:
                    context = f.read()
                    return self._tokenizer.encode(context)
            except UnicodeDecodeError:
                pass  


            try:
                with open(txt, "r", encoding="gbk") as f:
                    context = f.read()
                    return self._tokenizer.encode(context)
            except UnicodeDecodeError:
                pass


            try:
                with open(txt, "r", encoding="gb18030") as f:
                    context = f.read()
                    return self._tokenizer.encode(context)
            except UnicodeDecodeError:
                pass

            try:
                with open(txt, "r", encoding="gb2312") as f:
                    context = f.read()
                    return self._tokenizer.encode(context)

            except UnicodeDecodeError:
                tqdm.write(f"无法打开{txt_name}，非utf-8, gbk, gb2312, gb18030文件")
                return None
            
            except Exception as e:             
                # 捕获其他未预期的异常（可选，用于调试）
                tqdm.write(f"未知错误（{txt_name}）")
                return None


        token_ids = []
        for txt_file in tqdm(txt_files, desc="编码"):
            ids = text_encode(txt_file)
            if ids is not None:
                token_ids.extend(ids)
                # 每本小说后面添加一个EOS
                token_ids.extend([self._tokenizer.eos_id])


        dataset = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            dataset.append({"input": input_chunk, "target": target_chunk})

        # 划分数据集
        train_set, valid_set = train_test_split(
            dataset,
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

    tokenizer_t  = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    preprocessor_t = TextProcessor(tokenizer_t)
    preprocessor_t.processing()