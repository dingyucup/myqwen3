from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(usage="usage: main.py action")

    parser.add_argument(
        "action",
    choices=["train", "predict", "evaluate", "serve", "preprocess", "help"]
    )

    args = parser.parse_args()
    action = args.action

    match action:
        case "preprocess":
            from preprocess.tokenizer import Tokenizer
            from preprocess.text_processor import Preprocessor
            tokenizer = Tokenizer.from_tiktoken("gpt2")
            preprocessor = Preprocessor(tokenizer)
            preprocessor.processing()
        case "train":
            pass
        case "predict":
            pass
        case "help":
            print("pyton main.py [action]")
            print("="*5, "action", "="*5)
            print("preprocess: 数据预处理")
            print("train: 训练")
            print("evaluate: 评价")
            print("serve: 服务")
            print("predict: 预测")
            print("help: 帮助")
        case "evaluate":
            pass
        case "serve":
            pass
        case _:
            print("帮助命令: python main.py help")
