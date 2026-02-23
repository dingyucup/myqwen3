import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time

from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd


from src.configuration.config import *
from src.predict.predictor_kv_cache import CachedPredictor

def sample_rejected_data(
        question_file,
        answer_file,
        model,
        tokenizer,
        max_num_new_tokens=50,
        context_size=CONTEXT_LENGTH,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.2,
        sample_frac = 0.0001,
        random_state = 123,
        device=None
):
    # 加载数据集
    df = pd.read_json(
        str(question_file),
        lines=True,
        orient="records",
        encoding="utf-8"
    )
    # 删除空列
    df = df.drop(columns=["history", "response"])
    # 删除数值为NAN
    df = df.dropna(how="any")
    # 转为字典
    df = df.sample(frac=sample_frac, random_state=random_state).to_dict(orient="records")
    # 计算数据集长度
    data_len = len(df)
    print(f"总问题数据量：{data_len}")

    rejected_data = []
    count = 0
    for item in tqdm(df, "进度"):
        answer = CachedPredictor.generate_answer(
            question_text=item["question"],
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=max_num_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            # top_p=0.9,
            eos_id=tokenizer.eos_id,
            repetition_penalty=repetition_penalty,
            device=device
        )

        rejected_data.append({"prompt": item["question"], "rejected": answer["top_k"], "raw": item["answer"]})
        count += 1
        # 数据量大需要即时保存
        if count % 500 == 0:
            temp_file = str(DPO_RAW_DIR) + "/rejected_answer_" + str(count) + ".jsonl"
            pd.DataFrame(rejected_data).to_json(temp_file, orient="records", lines=True)


    pd.DataFrame(rejected_data).to_json(answer_file, orient="records", lines=True)
    return rejected_data


def sample_chosen_data(
        rejected_file,
        answer_file,
        # sample_frac=1.0,
        # random_state=123,

):
    # 加载数据集
    df = pd.read_json(
        str(rejected_file),
        lines=True,
        orient="records",
        encoding="utf-8"
    )
    # 删除空列
    # 删除数值为NAN
    df = df.dropna(how="any")
    # 转为字典
    # df = df.sample(frac=sample_frac, random_state=random_state).to_dict(orient="records")
    df = df.to_dict(orient="records")
    # 计算数据集长度
    data_len = len(df)
    print(f"总问题数据量：{data_len}")

    llm = ChatOllama(
        model="qwen3:latest",
        base_url= "http://127.0.0.1:11434/",
        temperature=0.1
    )

    chat_template = ChatPromptTemplate(
        [
            ("system", "你是一名数据标注优化员，负责根据给定的问题和原始答案，生成一个更优质、准确的回答。请确保：1. 回答字数严格控制在100字以内；2. 不使用任何表情符号；3. 若原答案有误，请予以更正；4. 回答应简洁、精炼、直接。"),
            ("human", "问题：{question}"),
            ("human", "原始答案：{answer}"),
        ]
    )


    str_output = StrOutputParser()

    chain = chat_template | llm | str_output

    chosen_data = []
    count = 0
    for item in tqdm(df[count:], "进度"):
        output = chain.invoke({"question": item["prompt"], "answer": item["raw"]})
        chosen_data.append(
            {"prompt": item["prompt"], "chosen": output, "rejected": item["rejected"]}
        )

        count += 1
        if count % 100 == 0:
            temp_file = str(DPO_RAW_DIR) + "/chosen_answer_" + str(count) + ".jsonl"
            pd.DataFrame(chosen_data).to_json(temp_file, orient="records", lines=True)
            time.sleep(120)


    pd.DataFrame(chosen_data).to_json(answer_file, orient="records", lines=True)
    return chosen_data

def promote_data_quality(
        data_file,
        output_file,
        model,
        tokenizer,
        max_num_new_tokens=50,
        context_size=CONTEXT_LENGTH,
        temperature=0.5,
        top_k=50,
        repetition_penalty=1.2,
        device=None
):
    # 删除空列
    # 删除数值为NAN
    df = pd.read_json(
        str(data_file),
        lines=True,
        orient="records",
        encoding="utf-8"
    )
    df = df.dropna(how="any")
    # 转为字典
    df = df.to_dict(orient="records")
    # 计算数据集长度
    data_len = len(df)
    print(f"总问题数据量：{data_len}")

    new_data_collection = []
    count = 0
    for item in tqdm(df, "进度"):
        answer = CachedPredictor.generate_answer(
            question_text=item["prompt"],
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=max_num_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            # top_p=0.9,
            eos_id=tokenizer.eos_id,
            repetition_penalty=repetition_penalty,
            device=device
        )

        new_data_collection.append(
            {
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": answer["top_k"],
            }
        )
        count += 1
        # 数据量大需要即时保存
        if count % 500 == 0:
            temp_file = str(DPO_RAW_DIR) + "/new_rejected_answer_" + str(count) + ".jsonl"
            pd.DataFrame(new_data_collection).to_json(temp_file, orient="records", lines=True)

    pd.DataFrame(new_data_collection).to_json(output_file, orient="records", lines=True)
    return new_data_collection




if __name__ == "__main__":
    import torch
    from src.preprocess.tokenizer import Tokenizer
    from src.model.qwen3_model import Qwen3Model
    from src.transformer.attention_kv_cache import add_kv_cache_to_model

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
    lora_model_t = Qwen3Model.from_lora_pretrained(LORA_MODELS_DIR/LORA_MODEL_NAME)
    lora_model_t = add_kv_cache_to_model(lora_model_t)
    lora_model_t.to(device_t)

    # sample_rejected_data(
    #     question_file=INSTRUCT_RAW_DIR/INSTRUCT_RAW_FILE,
    #     answer_file= DPO_RAW_DIR/DPO_REJECTED_GEN_FILE,
    #     model=lora_model_t,
    #     tokenizer=tokenizer_t,
    #     max_num_new_tokens=100,
    #     context_size=CONTEXT_LENGTH,
    #     temperature=1.5,
    #     top_k=50,
    #     repetition_penalty=1.18,
    #     sample_frac= 0.0001,
    #     random_state=123,
    #     device=device_t
    # )

    # result = sample_chosen_data(
    #     rejected_file=DPO_RAW_DIR/DPO_REJECTED_GEN_FILE,
    #     answer_file=DPO_RAW_DIR/DPO_CHOSEN_GEN_FILE
    # )

    res = promote_data_quality(
        data_file=DPO_RAW_DIR/"dpo_chosen_gen.jsonl",
        output_file=DPO_RAW_DIR/"dpo_promote_data.jsonl",
        model=lora_model_t,
        tokenizer=tokenizer_t,
        max_num_new_tokens=100,
        context_size=CONTEXT_LENGTH,
        temperature=0.5,
        top_k=100,
        repetition_penalty=1.0,
        device=device_t
    )

    print(res)



