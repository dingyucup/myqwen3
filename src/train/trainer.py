import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *

import torch
from torch import nn, optim
from dataclasses import dataclass
from tqdm import tqdm
import time

from src.model.qwen3_model import Qwen3Model
from src.preprocess.dataset import GPTDataset, creat_data_loader
from src.preprocess.tokenizer import Tokenizer
from src.predict.predictor import Predictor


@dataclass
class TrainConfig:
    models_dir: Path = MODELS_DIR
    logs_dir: Path =  LOGS_DIR
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    eval_freq = 500
    best_model_name = BEST_MODEL_NAME


train_config = TrainConfig()
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
            self,
            cfg: TrainConfig,
            model: Qwen3Model,
            optimizer: optim.Optimizer,
            loss: nn.CrossEntropyLoss,
            train_dataset: GPTDataset,
            valid_dataset: GPTDataset,
            device: torch.device
    ):
        self.config = cfg
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = device
        self.steps = 0
        self.min_valid_loss = float("inf")

    @classmethod
    def from_config(
            cls,
            model: Qwen3Model,
            train_cfg: TrainConfig = train_config,
            train_dataset_file = str(PREPROCESSED_DATA_DIR/TRAIN_DATA_FILE),
            valid_dataset_file = str(PREPROCESSED_DATA_DIR/VALID_DATA_FILE),
            device: torch.device = default_device
    ):
        # 模型加载
        model.to(device)

        # 优化器，AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )

        # 损失函数
        loss = nn.CrossEntropyLoss()

        # 加载数据集
        train_dataset = GPTDataset.from_json(train_dataset_file)
        valid_dataset = GPTDataset.from_json(valid_dataset_file)

        # print(model)

        return cls(
            cfg = train_cfg,
            model=model,
            optimizer=optimizer,
            loss = loss,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=device
        )

    def _calc_train_loss_batch(self, inputs, targets):
        self.steps += 1
        # 梯度清零
        self.optimizer.zero_grad()
        # 前向传播
        outputs = self.model(inputs)
        # 计算损失

        # loss_value = torch.nn.functional.cross_entropy(
        #     outputs.flatten(0, 1),
        #     targets.flatten()
        # )

        loss_value = self.loss(outputs.transpose(-1, -2), targets)
        # 反向传播
        loss_value.backward()
        # 梯度更新
        self.optimizer.step()

        return loss_value.item()

    # 计算验证损失
    def _calc_valid_loss(self, valid_data_loader):
        self.model.eval()
        with torch.no_grad():
            valid_loss_accumulate = 0
            count = 0
            for inputs, targets in valid_data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss_value = self.loss(outputs.transpose(-1, -2), targets)
                valid_loss_accumulate += loss_value * inputs.shape[0]
                count += inputs.shape[0]

            return valid_loss_accumulate / count


    def train(self):
        if (self.config.models_dir/self.config.best_model_name).exists():
            try:
                tqdm.write(f"加载预训练模型...")
                checkpoint = torch.load(self.config.models_dir/self.config.best_model_name)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.steps = checkpoint["steps"]
                self.min_valid_loss = checkpoint["min_valid_loss"]
                tqdm.write(f"预训练模型加载成功，开始训练...")
            except Exception as e:
                tqdm.write(f"加载预训练模型失败: {e}")
                tqdm.write("将从头开始训练...")
        else:
            tqdm.write("未找到预训练模型，将从头开始训练...")

        print("倒计时30秒")
        for _ in tqdm(range(30), desc="启动前等待", bar_format="{desc}: {remaining}s"):
            time.sleep(1)
        print("训练开始...")


        self.model.train()
        # 准备数据加载器
        train_data_loader = creat_data_loader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )

        valid_data_loader = creat_data_loader(
            dataset=self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )

        # min_loss = float("inf")
        train_loss_accumulate = 0.0
        count = 0
        writer = SummaryWriter(
            log_dir=str(self.config.logs_dir / time.strftime("%Y-%m-%d_%H-%M-%S"))
        )

        # 测试用例
        tokenizer = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)
        text = "我深爱着你！"
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(default_device)
        # 初始文本生成测试
        result = Predictor.generate(
            model=self.model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_num_new_tokens=25,
            context_size=CONTEXT_LENGTH
        )
        print("Output text:\n", result)

        for epoch in range(self.config.epochs):
            for inputs, targets in tqdm(train_data_loader, desc="train"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                train_loss_batch = self._calc_train_loss_batch(inputs, targets)
                train_loss_accumulate += train_loss_batch * inputs.shape[0]
                count += inputs.shape[0]
                if self.steps % self.config.eval_freq == 0:
                    train_loss = train_loss_accumulate / count
                    print("computing validation loss, please wait a minute~")
                    valid_loss = self._calc_valid_loss(valid_data_loader)
                    tqdm.write(f"EPOCH {epoch + 1}|STEP {self.steps}|train_loss:{train_loss:.6f}|valid_loss: {valid_loss:.6f}")
                    writer.add_scalar("train_loss", train_loss, self.steps)
                    writer.add_scalar("valid_loss", valid_loss, self.steps)

                    if valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        # 保存模型参数
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "steps": self.steps,
                            "min_valid_loss": self.min_valid_loss
                        },
                            str(self.config.models_dir/self.config.best_model_name)
                        )
                        print("the newest model have been saved.")

                        # 文本生成测试
                        result = Predictor.generate(
                            model=self.model,
                            tokenizer=tokenizer,
                            input_ids=input_ids,
                            max_num_new_tokens=25,
                            context_size=CONTEXT_LENGTH
                        )
                        print("Output text:\n", result)

                    train_loss_accumulate = 0
                    count = 0



if __name__ == "__main__":
    torch.manual_seed(42)
    model_t = Qwen3Model.from_config()
    trainer = Trainer.from_config(model_t, train_config)

    trainer.train()







