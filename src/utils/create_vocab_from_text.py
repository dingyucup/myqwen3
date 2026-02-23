import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from collections import Counter
import jieba
from tqdm import tqdm

from src.configuration.config import *


def create_vocab_from_text(
        txt_dir=TXT_FOR_VOCAB_DIR,
        dict_vocab_file = PREPROCESSED_DATA_DIR / "XDHYCD.txt",
        vocab_file=MODELS_DIR/VOCAB_FILE
):
    # 加载词典
    jieba.load_userdict(str(dict_vocab_file))
    with open(str(dict_vocab_file), "r", encoding="utf-8") as dict_file:
        dict_words = dict_file.read()
        zh_set = set(dict_words.split("\n"))

    # 罗列所有小说
    txt_files = [f.name for f in Path(txt_dir).glob("*.txt")]

    # 从小说中提取分词
    def words_from_txt(txt_name):
        words = list()
        txt = str(Path(txt_dir) / txt_name)
        try:
            with open(txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    # 精准模式分词
                    words += jieba.lcut(line.strip(), cut_all=False)
        except UnicodeDecodeError:
            tqdm.write(f"无法打开{txt_name}，非utf-8编码文件")
            return words

        return words

    # 查字典，如果字典中没有，按字切分
    def consult_zh_dict(word_list: list[str]) -> list[str]:
        nonlocal zh_set
        word_list_extra = []
        for word in word_list:
            # 如果字典中查不到，进行字符级分词
            if word in zh_set:
                word_list_extra.append(word)
            else:
                word_list_extra.extend(char for char in word if char.strip())

        return word_list_extra

    vocab_counter = Counter()

    for txt_file in tqdm(txt_files, desc="正在处理txt文件"):
        words_initial = words_from_txt(txt_file)
        words_final = consult_zh_dict(words_initial)
        vocab_counter.update(words_final)

    sorted_words_list = [word for word, count in vocab_counter.most_common()]

    save_path = str(vocab_file)
    with open(save_path, "w", encoding="utf-8") as writer:
        writer.write("\n".join(sorted_words_list))

    print(f"词表建立完成，保存至：{save_path}")

if __name__ == "__main__":
    create_vocab_from_text()