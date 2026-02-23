import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *

import re

from tqdm import tqdm

BRACKET_PATTERN = re.compile(r'【([^】]+)】')
PAREN_PATTERN = re.compile(r'[（\(][^）\)]*[）\)]')

def create_vocab_from_dict(
        raw_dict_file = RAW_DATA_DIR/"XDHYCD7th.txt",
        output_file = PREPROCESSED_DATA_DIR/"XDHYCD.txt"
):
    with open(str(raw_dict_file), 'r', encoding='utf-8') as raw_file:
        lines = raw_file.readlines()

    words = set()
    for line in tqdm(lines, desc="提取词汇"):
        match = BRACKET_PATTERN.match(line)
        if match:
            word = PAREN_PATTERN.sub('', match.group(1)).strip()
            if word:
                words.add(word)

    # 排序并写入
    with open(str(output_file), 'w', encoding='utf-8') as writer:
        words = sorted(list(words))
        writer.write("\n".join(words))

if __name__ == '__main__':
    create_vocab_from_dict()