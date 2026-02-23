import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jieba

from src.configuration.config import *

class Tokenizer:
    def __init__(
            self,
            vocab_list: list[str],
            pad_token = PAD,
            unk_token = UNK,
            eos_token = EOS
    ):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.id2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.word2id = {word: idx for idx, word in enumerate(vocab_list)}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        # self.id2word[-100] = pad_token
        # self.word2id[pad_token] = -100

        if self.unk_token not in self.word2id.keys():
            raise KeyError("unk_token未设置")

        if self.eos_token not in self.word2id.keys():
            raise KeyError("eos_token未设置")

        self.unk_id = self.word2id.get(self.unk_token)
        self.pad_id = self.word2id.get(self.pad_token)
        self.eos_id = self.word2id.get(self.eos_token)

    @classmethod
    def from_vocab(
            cls,
            vocab_file = MODELS_DIR/VOCAB_FILE,
            special_char_set = None
    ):
        # 读取词表
        try:
            with open(str(vocab_file), "r", encoding="utf-8") as vf:
                words = vf.read()
                vocab_list = [word for word in words.split("\n") if word.strip()]
        except FileNotFoundError:
            print("没有找到词表文件")


        # 给词表添加空白字符、换行符以及制表符
        vocab_list = ["\n"] + vocab_list + [" ", "\t"]
        if special_char_set is not None:
            special_char_set.difference_update(vocab_list)
            vocab_list = vocab_list + sorted(list(special_char_set))

        return cls(vocab_list)

    def tokenize(self, text: str):
        # jieba分词器加载词表
        for word in self.vocab_list:
            jieba.add_word(word=word)

        tokens_initial = jieba.lcut(text)
        tokens_list = []
        for token in tokens_initial:
            # 如果词表中中查不到，进行字符级分词
            if token in self.word2id.keys():
                tokens_list.append(token)
            else:
                tokens_list.extend(char for char in token if char.strip())

        return  tokens_list

    def encode(self, text: str):
        tokens_list = self.tokenize(text)
        return [self.word2id.get(word, self.unk_id) for word in tokens_list]

    def decode(self, token_ids: list[int] | int):
        if isinstance(token_ids, int):
            return self.id2word.get(token_ids)

        return "".join([self.id2word.get(token_id) for token_id in token_ids])

if __name__ == "__main__":
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    print(tokenizer_t.tokenize("我爱人工智能 __PAD__"))
    word_ids = tokenizer_t.encode("我爱人工智能 __PAD__")
    print(word_ids)
    print(tokenizer_t.decode(word_ids))
