import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.predict.predictor_steam import StreamPredictor
from src.model.qwen3_model import Qwen3Model
from src.configuration.config import *
from src.preprocess.tokenizer import Tokenizer
from src.transformer.attention_kv_cache import add_kv_cache_to_model

def ai_assistant(
        model: Qwen3Model,
        tokenizer: Tokenizer,
        device: torch.device
):
    model = add_kv_cache_to_model(model)
    print("你好！我是知识问答助手，请你提问，如需退出请输入 exit。")
    while True:
        user_input = input("问题 >>>")
        question = user_input.strip().replace("?", "")
        if question == "exit":
            break

        print("回答 >>>", end="")
        StreamPredictor.generate_answer_stream(
            question_text=question,
            model=model,
            tokenizer=tokenizer,
            max_num_new_tokens=50,
            context_size=CONTEXT_LENGTH,
            # temperature=0.3,
            top_p=0.95,
            eos_id=tokenizer.eos_id,
            device=device,
            repetition_penalty=1.2
        )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(123)
    # lora_model = Qwen3Model.from_pretrained(LORA_MODELS_DIR/MERGED_LORA_MODEL_NAME)
    lora_model = Qwen3Model.from_pretrained(MODELS_DIR/"dpo_best_model_sft.pt")
    # lora_model = Qwen3Model.from_lora_pretrained(MODELS_DIR/"dpo_best_model_with_lora_beta_0_0_5.pt")
    lora_model.to(device)

    tokenizer_t = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)

    ai_assistant(model=lora_model, tokenizer=tokenizer_t, device=device)


if __name__ == "__main__":
    main()



