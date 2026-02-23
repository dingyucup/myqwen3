import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.model.qwen3_model import Qwen3Model, Qwen3Config
from src.preprocess.tokenizer import Tokenizer
from src.preprocess.instruct_processor import format_input
from src.transformer.attention_kv_cache import reset_kv_cache_in_model


class StreamPredictor:
    @classmethod
    def repetition_penalty_func(cls, input_ids, logits, penalty_factor):
        if penalty_factor != 1.0:
            for i in range(input_ids.size(0)):  # 遍历每个样本（batch）
                unique_tokens = set(input_ids[i].tolist())  # 当前已生成的所有 token
                for token_id in unique_tokens:
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= penalty_factor
                    else:
                        logits[i, token_id] *= penalty_factor

        return logits


    @classmethod
    @torch.inference_mode()
    def _generate_greedy(
            cls,
            input_ids,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            repetition_penalty = 1.0,
            eos_id = None,
    ):
        reset_kv_cache_in_model(model)
        # 预填充阶段：处理所有提示token
        prompt_length = input_ids.shape[1]
        if prompt_length > 1:
            # 一次性处理提示词的token, 除了最后一个来填充KV缓存
           _ = model(input_ids[:, :-1])

        last_token_id = input_ids[:, -1:]

        max_num_new_tokens = min(max_num_new_tokens + prompt_length, context_size)

        generated_tokens = []  # 存储生成的token ids
        generated_text = []  # 存储生成的文本

        # 自回归生成
        model.eval()
        for i in range(max_num_new_tokens):
            current_input = last_token_id

            with torch.no_grad():
                logits = model(current_input)  # 对整个token串中每个token作一个下一词的概率预测

            # 取出对文本序列中最后一个token作下一个词在词表中的分布概率
            next_token_id_logits = logits[:, -1, :]

            # 重复性惩罚（需要完整历史）
            all_input_ids = torch.cat(
                [input_ids] + generated_tokens, dim=1
            ) if generated_tokens else input_ids

            # 重复性惩罚
            next_token_id_logits = cls.repetition_penalty_func(
                input_ids=all_input_ids,
                logits=next_token_id_logits,
                penalty_factor=repetition_penalty)

            # 概率分布归一化
            next_token_id_probability = torch.softmax(next_token_id_logits, dim=-1)
            # 贪心解码
            next_id = torch.argmax(next_token_id_probability, dim=-1, keepdim=True)
            next_token = tokenizer.decode(next_id.item())

            if next_id == eos_id:
                break
            else:
                yield next_token
                # 把预测出来的下一个词，添加到原来的序列中
                generated_tokens.append(next_id)
                generated_text.append(next_token)
                last_token_id = next_id

    @classmethod
    def generate_greedy_stream(
            cls,
            prompt_text,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            print_prompt_text = True,
            repetition_penalty=1.0,
            eos_id=None,
            device = torch.device("cpu")
    ):
        input_ids = tokenizer_t.encode(prompt_text)
        input_ids = torch.tensor([input_ids]).to(device)

        if print_prompt_text:
            print(f"{prompt_text}", end="")

        for token in cls._generate_greedy(
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=max_num_new_tokens,
            context_size=context_size,
            repetition_penalty=repetition_penalty,
            eos_id=eos_id,
        ):
            print(token, end="", flush=True)

        print()


    @classmethod
    @torch.inference_mode()
    def _generate_top_k(
            cls,
            input_ids,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            temperature:float = 0.0,
            top_k = None,
            eos_id = None,
            repetition_penalty = 1.0
    ):
        reset_kv_cache_in_model(model)
        # 预填充阶段：处理所有提示token
        prompt_length = input_ids.shape[1]
        if prompt_length > 1:
            # 一次性处理所有提示token来填充KV缓存
           _ = model(input_ids[:, :-1])

        last_token_id = input_ids[:, -1:]

        max_num_new_tokens = min(max_num_new_tokens + prompt_length, context_size)

        generated_tokens = []  # 存储生成的token ids
        generated_text = []  # 存储生成的文本

        # 自回归生成
        model.eval()
        for i in range(max_num_new_tokens):

            current_input = last_token_id  # 上次生成的token

            with torch.no_grad():
                logits = model(current_input)  # 对整个token串中每个token作一个下一词的概率预测
                # 取出对文本序列中最后一个token作下一个词在词表中的分布概率
                logits = logits[:, -1, :]

                all_input_ids = torch.cat(
                    [input_ids] + generated_tokens, dim=1
                ) if generated_tokens else input_ids

                # 增加重复性惩罚
                logits = cls.repetition_penalty_func(
                    input_ids= all_input_ids,
                    logits=logits,
                    penalty_factor=repetition_penalty
                )

                # top-k采样
                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    # 找到k采样中的最小概率值
                    min_val = top_logits[:, -1]
                    # 概率小于min_val统统赋值为-inf,以便在softmax计算时变为0
                    bools: torch.Tensor = logits < min_val
                    logits = torch.where(
                        condition=bools,
                        input=torch.tensor(float("-inf")).to(logits.device),
                        other=logits
                    )

                # 温度缩放
                if temperature > 0.0:
                    logits = logits / temperature
                    # 重新计算概率
                    probs = torch.softmax(logits, dim=-1)
                    # 采样
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                next_token = tokenizer.decode(next_id.item())

                if next_id == eos_id:
                    break
                else:
                    yield next_token
                    # 把预测出来的下一个词，添加到原来的序列中
                    generated_tokens.append(next_id)
                    generated_text.append(next_token)
                    last_token_id = next_id

    @classmethod
    def generate_top_k_stream(
            cls,
            prompt_text,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            temperature: float = 0.0,
            top_k=None,
            eos_id=None,
            repetition_penalty=1.0,
            print_prompt_text = True,
            device = None
    ):

        input_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids]).to(device)

        if print_prompt_text:
            print(f"{prompt_text}", end="")

        for token in cls._generate_top_k(
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=max_num_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
            repetition_penalty=repetition_penalty
        ):
            print(token, end="", flush=True)

        print()

    @classmethod
    @torch.inference_mode()
    def _generate_top_p(
            cls,
            input_ids,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            temperature: float = 0.0,
            top_p=None,
            eos_id=None,
            repetition_penalty = 1.0
    ):
        reset_kv_cache_in_model(model)
        # 预填充阶段：处理所有提示token
        prompt_length = input_ids.shape[1]
        if prompt_length > 1:
            # 一次性处理所有提示token来填充KV缓存
           _ = model(input_ids[:, :-1])

        last_token_id = input_ids[:, -1:]

        max_num_new_tokens = min(max_num_new_tokens + prompt_length, context_size)

        generated_tokens = []  # 存储生成的token ids
        generated_text = []  # 存储生成的文本

        # 自回归生成
        model.eval()
        for i in range(max_num_new_tokens):

            current_input = last_token_id  # 上次生成的token

            with torch.no_grad():
                logits = model(current_input)  # 对整个token串中每个token作一个下一词的概率预测
                # 取出对文本序列中最后一个token作下一个词在词表中的分布概率
                logits = logits[:, -1, :]

                # 重复性惩罚（需要完整历史）
                all_input_ids = torch.cat(
                    [input_ids] + generated_tokens, dim=1
                ) if generated_tokens else input_ids

                # 增加重复性惩罚
                logits = cls.repetition_penalty_func(
                    input_ids= all_input_ids,
                    logits=logits,
                    penalty_factor=repetition_penalty
                )

                # 温度缩放
                if temperature > 0.0:
                    logits = logits / temperature

                # 计算概率
                probs = torch.softmax(logits, dim=-1)

                # top-p (nucleus) 采样
                if top_p is not None and top_p > 0:
                    # 对概率进行降序排序
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

                    # 计算累积概率
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # 创建掩码：移除累积概率超过top_p的token
                    # 保留累积概率小于等于top_p的token
                    sorted_mask = cumulative_probs <= top_p

                    # 确保至少有一个token被选中（如果第一个token的概率就大于top_p，也要保留它）
                    # 将第一个token的掩码设为True
                    sorted_mask[:, 0] = True

                    # 将需要移除的token的概率设为0
                    sorted_probs = sorted_probs * sorted_mask.float()

                    # 重新归一化概率
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                    # 将排序后的概率重新映射回原始顺序
                    probs = torch.zeros_like(probs)
                    probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)


                # 采样或贪婪解码
                if temperature > 0.0:
                    # 使用修改后的概率分布进行采样
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪婪解码：选择概率最高的token
                    next_id = torch.argmax(probs, dim=-1, keepdim=True)

                next_token = tokenizer.decode(next_id.item())

                if next_id == eos_id:
                    break
                else:
                    yield next_token
                    # 保存生成的token
                    generated_tokens.append(next_id)
                    generated_text.append(next_token)
                    # 把预测出来的下一个词，添加到原来的序列中
                    last_token_id = next_id

    @classmethod
    def generate_top_p_stream(
            cls,
            prompt_text,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            temperature: float = 0.0,
            top_p=None,
            eos_id=None,
            repetition_penalty=1.0,
            print_prompt_text = True,
            device = None
    ):
        input_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([input_ids]).to(device)

        if print_prompt_text:
            print(f"{prompt_text}", end="")

        for token in cls._generate_top_p(
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=max_num_new_tokens,
            context_size=context_size,
            temperature=temperature,
            top_p=top_p,
            eos_id=eos_id,
            repetition_penalty=repetition_penalty,
        ):
            print(token, end="", flush=True)

        print()


    @classmethod
    def generate_answer_stream(
            cls,
            question_text,
            model: Qwen3Model,
            tokenizer: Tokenizer,
            max_num_new_tokens: int,
            context_size: int,
            temperature: float = 0.0,
            top_k=None,
            top_p = None,
            eos_id=None,
            repetition_penalty = 1.0,
            device = None
    ):
        instruct_dict = {"question": question_text, "answer": ""}
        instruct_text = format_input(instruct_dict)

        if top_k is not None:
            cls.generate_top_k_stream(
                prompt_text=instruct_text,
                model=model,
                tokenizer=tokenizer,
                max_num_new_tokens=max_num_new_tokens,
                context_size=context_size,
                top_k=top_k,
                temperature=temperature,
                eos_id=eos_id,
                repetition_penalty=repetition_penalty,
                print_prompt_text = False,
                device=device
            )



        if top_p is not None:
            cls.generate_top_p_stream(
                prompt_text=instruct_text,
                model=model,
                tokenizer=tokenizer,
                max_num_new_tokens=max_num_new_tokens,
                context_size=context_size,
                top_p=top_p,
                temperature=temperature,
                eos_id=eos_id,
                repetition_penalty=repetition_penalty,
                device=device,
                print_prompt_text=False
            )


        if top_p is None and top_k is None:
            cls.generate_greedy_stream(
                prompt_text=instruct_text,
                model=model,
                tokenizer=tokenizer,
                max_num_new_tokens=max_num_new_tokens,
                context_size=context_size,
                repetition_penalty=repetition_penalty,
                eos_id=eos_id,
                print_prompt_text=False,
                device=device
            )



if __name__ == "__main__":
    # torch.manual_seed(123)
    # text_t = "我深情地望着她"
    text_t = "清晨的阳光洒在海面上，和煦的海风轻抚，我们紧紧依偎"
    # torch.manual_seed(1230)
    # text_t = "我深情地望着她"
    # torch.manual_seed(1999)
    # text_t = "“我喜欢你的歌声”，我望着她"

    # torch.manual_seed(1999)
    # text_t = "我深爱着你"
    # text_t = "我望着她"
    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.configuration.config import *
    from src.transformer.attention_kv_cache import add_kv_cache_to_model
    from src.model.qwen3_model import Qwen3Model

    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)

    model_t = Qwen3Model.from_pretrained(BASE_MODEL_DIR/BASE_MODEL_NAME)
    model_t = add_kv_cache_to_model(model_t)
    model_t.to(device_t)

    print(model_t)

    print("[贪心采样]:", end="")
    StreamPredictor.generate_greedy_stream(
        prompt_text=text_t,
        model=model_t,
        tokenizer=tokenizer_t,
        max_num_new_tokens=128,
        context_size=CONTEXT_LENGTH,
        eos_id=tokenizer_t.eos_id,
        repetition_penalty=1.18,
        device=device_t
    )


    print("[top-k采样]", end=":")
    StreamPredictor.generate_top_k_stream(
        prompt_text=text_t,
        model=model_t,
        tokenizer=tokenizer_t,
        max_num_new_tokens=128,
        context_size=CONTEXT_LENGTH,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.18,
        device=device_t
    )

    print("[top-p采样]", end=":")
    StreamPredictor.generate_top_p_stream(
        prompt_text=text_t,
        model=model_t,
        tokenizer=tokenizer_t,
        max_num_new_tokens=128,
        context_size=CONTEXT_LENGTH,
        top_p=0.9,
        temperature=0.3,
        repetition_penalty=1.18,
        device=device_t
    )



    lora_model_t = Qwen3Model.from_lora_pretrained(LORA_MODELS_DIR/LORA_MODEL_NAME)
    lora_model_t = add_kv_cache_to_model(lora_model_t)
    lora_model_t.to(device_t)

    question_t  = "太阳系有多少行星？"
    StreamPredictor.generate_answer_stream(
        question_text=question_t,
        model=lora_model_t,
        tokenizer=tokenizer_t,
        max_num_new_tokens=50,
        context_size = CONTEXT_LENGTH,
        temperature=0.1,
        top_k=50,
        # top_p=0.9,
        eos_id=tokenizer_t.eos_id,
        repetition_penalty=1.2,
        device=device_t
    )


