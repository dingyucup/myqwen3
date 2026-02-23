from pathlib import Path
############ 目录相关 ############
#主目录
ROOT_DIR = Path(__file__).parent.parent.parent
# 数据目录
DATA_DIR = ROOT_DIR/"data"
# 原始数据目录
RAW_DATA_DIR = DATA_DIR/"raw"
# 预处理后数据目录
PREPROCESSED_DATA_DIR = DATA_DIR/"preprocessed"
# 模型相关目录
MODELS_DIR = ROOT_DIR/"models"

############ 预训练相关 ############
# 基础模型目录
BASE_MODEL_DIR = MODELS_DIR/"base"
# 词表文件
VOCAB_FILE = "vocab.txt"
# 处理后的数据文件
TRAIN_DATA_FILE = "train.jsonl"
VALID_DATA_FILE = "valid.jsonl"
# 日志目录
LOGS_DIR = ROOT_DIR/"logs"
# 创建词表的TXT文件目录
TXT_FOR_VOCAB_DIR = RAW_DATA_DIR/"txt_files_for_vocab"
# 小说TXT文件目录
NOVELS_TXT_DIR = RAW_DATA_DIR/"novels_txt"
# 保存最佳模型
BEST_MODEL_NAME = "best_model.pt"
# 从最佳模型中提取基础模型（去除优化器器参数）
BASE_MODEL_NAME = "best_base_model.pt"

############ SFT相关(采用LoRA) ############

# LoRA模型目录
LORA_MODELS_DIR = MODELS_DIR/"lora"
# 指令集原始文件目录
INSTRUCT_RAW_DIR = RAW_DATA_DIR/"instruction"
# 指令集处理后文件目录
INSTRUCT_PROCESSED_DIR = PREPROCESSED_DATA_DIR/"instruction"
# 问答对指令集原始文件
INSTRUCT_RAW_FILE = "question_and_answer.jsonl"
# LoRA指令微调模型
LORA_MODEL_NAME = "best_lora_model.pt"
# 合并参数LoRA参数后的模型
MERGED_LORA_MODEL_NAME = "merged_lora_model.pt"

# LoRA模型参数
RANK = 16
ALPHA = 32

############ 对齐相关(采用DPO) ############
# DPO偏好原始数据集目录
DPO_RAW_DIR = RAW_DATA_DIR/"dpo"
# DPO偏好处理后数据集目录
DPO_PROCESSED_DIR = PREPROCESSED_DATA_DIR/"dpo"
# DPO中优质回答生成文件（包含了劣质回答）
DPO_CHOSEN_GEN_FILE = "dpo_chosen_gen.jsonl"
# DPO中劣质回答生成文件
DPO_REJECTED_GEN_FILE = "dpo_rejected_gen.jsonl"
# 总的偏好数据集
DPO_PREFERENCE_RAW_DATA_FILE = "dpo_promote_data.jsonl"
# 输出DPO模型的名字
DPO_BEST_MODEL_WITH_LORA = "dpo_best_model_with_lora.pt"
DPO_BEST_MODEL_SFT = "dpo_best_model_sft.pt"

# 计算损失函数时用到的beta值
DPO_BETA = 0.5

############ 模型相关 ############
# 最长序列大小，用于序列截断
SEQ_LENGTH = 1024
# 文本序列长度，用于构建因果注意力的掩码
CONTEXT_LENGTH = 1024
# 滑动距离，处理文本时的跨度，一般设置和CONTEXT_LENGTH相同
STRIDE = 1024
# 训练集比例
TRAIN_SET_RATIO = 0.9

# 词向量维度
EMB_DIM = 512
# 头向量的维度
HEAD_DIM = 64
# 注意力头的数量
N_HEADS = 8
# 层数
N_LAYERS = 12
# 分组查询注意力中需要给多个头进行KV分组，所分的组数
N_KV_GROUPS = 2
# 专家总数量
N_EXPERTS = 16
# 邀请几位“顶级”专家处理一个Token
NUM_EXPERTS_PER_TOKEN = 2
# 专家之间传递的向量维度
MOE_INTERMEDIATE_SIZE = 1024
# Key和Query投影后的向量是否需要归一化
QK_NORM = True

# 特殊字符
UNK = "__UNK__"
EOS = "__EOS__"
PAD = "__PAD__" # 记得在损失函数中设置ignore_index
# 特殊字符集合
SPECIAL_CHAR_SET = {UNK, EOS, PAD}

############ 训练相关 ############
# 是否加载预训练模型
RESUME_MODEL = True
# 是否加载优化器
RESUME_OPTIMIZER = False
#批次大小
BATCH_SIZE = 2
# 学习率
LEARNING_RATE = 1e-5
# 学习率预热步数占总步数的比例
WARMUP_STEPS_RATIO = 0.1
# 权值衰减
WEIGHT_DECAY = 0.1
# 轮次数
EPOCHS = 2
# 最大步数
MAX_STEPS = 5000
# 多少步评价一次
EVAL_STEPS = 50


if __name__ == "__main__":
    print(ROOT_DIR)
