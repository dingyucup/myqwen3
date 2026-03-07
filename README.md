# myqwen3

## 项目简介

本项目是一个从零实现的轻量级类语言模型（参数约为0.3B），涵盖了从数据预处理、模型构建、训练到推理的完整流程。项目旨在帮助开发者深入理解现代大语言模型的内部原理，同时提供一个可扩展的代码框架，用于研究和实验。

### 主要特点

- **完整的模型架构**：实现了分组查询注意力（GQA）、旋转位置编码（RoPE）、RMSNorm、混合专家（MoE）前馈网络等先进结构。
- **丰富的训练范式**：
  - 预训练（Pretraining）
  - 指令微调（Instruction Fine-tuning）
  - 人类偏好对齐（DPO，Direct Preference Optimization）
  - LoRA 高效微调（支持 MoE 层的选择性跳过）
- **高效的推理实现**：
  - 支持 KV Cache 加速推理
  - 支持流式输出（Streaming）
- **自定义分词器**：可利用小说文本或现代汉语词典构建词表，灵活适配中文场景。
- **模块化设计**：各组件（注意力、前馈、归一化、位置编码等）独立封装，易于替换和扩展。
- **训练脚本齐全**：包括混合精度训练、模型保存与加载、LoRA 合并、DPO 数据构建等实用工具。

通过本项目，您可以学习到如何从零构建一个可用于实际对话的生成式语言模型，并掌握主流训练技术和优化手段。

---

## 创建虚拟环境

```bash
conda create -n myqwen3 python=3.12
```

## 激活虚拟环境

```bash
conda activate myqwen3
```

## 安装依赖

### 安装 pytorch

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 安装其余依赖

```bash
pip install jieba transformers datasets tensorboard tqdm jupyter pandas scikit-learn langchain langchain-community
```

## 项目结构

```bash
├── README.md
├── data 数据集
│   ├── preprocessed 预处理后的数据
│   │   ├──train.jsonl 预训练的训练数据集
│   │   ├── valid.jsonl 预训练的验证数据集
│   │   ├── XDHYCD.txt 现代汉语词典
│   │   ├── dpo 人类偏好对齐用到的数据集目录
│   │   └── instruction 指令微调用到的数据集目录
│   └── raw 原始数据集
│       ├── dpo
│       │   ├── dpo_chosen_gen.jsonl
│       │   ├── dpo_preference_raw_data.jsonl
│       │   ├── dpo_promote_data.jsonl
│       │   └── dpo_rejected_gen.jsonl
│       ├── instruction
│       │   └── question_and_answer.jsonl
│       ├── novels_text 小说集
│       └── txt_files_for_vocab 构建词表用的小说集
├── docs 文档
├── logs 日志
├── models 模型目录
│   ├── dpo_best_model_sft.pt **经过预训练-指令微调-对齐后的模型**❤️
│   ├── base 预训练好的模型目录
│   ├── lora 采用lora微调的模型目录
│   └── vocab.txt **词表**
└── src
    ├── __init__.py
    ├── configuration
    │   ├── __init__.py
    │   └── config.py **配置文件**
    ├── lora
    │   ├── __init__.py
    │   └── lora.py LoRA层
    ├── main.py 模型调用的主入口
    ├── model
    │   ├── __init__.py
    │   └── qwen3_model.py 模型主题架构
    ├── predict
    │   ├── __init__.py
    │   ├── predictor.py 模型推理
    │   ├── predictor_kv_cache.py 含有KV Cache的推理
    │   └── predictor_steam.py 流式输出的推理
    ├── preprocess
    │   ├── __init__.py
    │   ├── dataset.py 加载自定义的数据集类
    │   ├── dpo_data_creator.py 创建DPO数据集
    │   ├── dpo_dataset.py 加载DPO数据集
    │   ├── dpo_preference_processor.py DPO数据采集脚本
    │   ├── instruct_processor.py 指令对数据处理脚本
    │   ├── text_processor.py 小说文本处理脚本
    │   └── tokenizer.py **分词器**
    ├── train
    │   ├── __init__.py
    │   ├── trainer.py 模型训练脚本
    │   ├── trainer_amp.py 混合精度的模型训练脚本
    │   ├── trainer_dpo.py DPO训练脚本
    │   └── trainer_lora.py 含有LoRA的模型微调训练脚本
    ├── transformer
    │   ├── __init__.py
    │   ├── attention.py **分组查询注意力类**
    │   ├── attention_kv_cache.py 含有KV Cache的注意力类
    │   ├── feed_forward.py 前馈网络
    │   ├── layer_norm.py 归一化层
    │   ├── rope.py **旋转位置编码**
    └── utils
        ├── __init__.py
        ├── create_vocab_from_dict.py 通过《现代汉语词典》创建词表
        ├── create_vocab_from_text.py 通过小说集创建词表
        ├── download_datasets.py 
        ├── extract_model.py 将模型和优化器分离
        └── merge_and_save_lora_model.py 合并LoRA
```



## 模型权重与训练集

通过网盘分享的文件：myqwen3
链接: https://pan.baidu.com/s/1TcOfNJ2j3dHi7kWrRl3pUw?pwd=kmi8 提取码: kmi8 

## 测试效果
```txt

你好！我是知识问答助手，请你提问，如需退出请输入 exit。
问题 > 你是谁
回答 > 我是AI模型，没有实体存在
问题 > 中国的首都在哪里？
回答 > 中国的首都位于北京市。
问题 > 北京是中国的首都吗？
回答 > 是的，北京市位于中国北方，属于华北平原的中央地带。
问题 > 上海是中国的首都吗？
回答 > 不是，上海是直辖市，不属于任何省份，直接由中央政府管辖。
问题 > 谁是李冰冰？
回答 > 李冰冰是中国著名女演员和制片人，以出演多部知名电影而闻名，如《雪花秘扇》、《生化危机6：终章》等。
问题 > 谁是刘德华？
回答 > 刘德华是中国香港著名歌手、演员和制片人，以其多才多艺著称，在音乐、电影及电视剧领域均有出色表现
问题 > 如何制作宫保鸡丁？
回答 > 准备鸡肉切丁、花生米、干辣椒、葱姜蒜等调料；将鸡肉切块备用；热油锅下花椒爆香，加入干辣椒和花椒炒出红油；放入鸡丁翻炒至变色，加生抽、老抽调色；倒入鸡丁快速翻炒均匀即可。
问题 > 如何制作鱼香肉丝？
回答 > 准备材料：鱼香肉丝100克、葱姜蒜2大勺、料酒适量；调料：盐、鸡精、胡椒粉、香油、葱花；调味料：生抽、老抽、糖、醋、水淀粉和香油。步骤：1.将鱼香油与姜蒜混合后加入花椒炒出香味；2.加入适量清水烧开后放入鱼香肉丝；3.加水煮沸后转小火慢炖至熟透；4.撒上葱花即可。
问题 > 如何制作面包？
回答 > 准备面粉、酵母、水和盐；将面团发酵至两倍大；烘烤约30分钟；热锅冷油下入烘焙好的面团翻炒至表面微焦；加入适量糖和水调味即可。
问题 > 介绍一下辛弃疾
回答 > 辛弃疾是宋代著名词人，以其豪放风格和爱国情怀著称，代表作有《破阵子·为陈同甫赋壮词以寄之》等。
问题 > 洗衣机的用途是什么？
回答 > 清洗衣物、去除污渍、保持卫生和清洁衣物。
```