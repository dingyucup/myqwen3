import sys
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *

import torch
from torch import nn, optim
from torch.amp import GradScaler
from dataclasses import dataclass
from tqdm import tqdm
import time

from src.model.qwen3_model import Qwen3Model
from src.preprocess.dataset import GPTDataset, create_data_loader, create_instruct_data_loader
from src.preprocess.tokenizer import Tokenizer
from src.predict.predictor import Predictor


@dataclass
class TrainConfig:
    models_dir: Path = MODELS_DIR
    logs_dir: Path =  LOGS_DIR
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    max_steps = MAX_STEPS
    learning_rate = LEARNING_RATE
    warmup_steps_ratio = WARMUP_STEPS_RATIO
    weight_decay = WEIGHT_DECAY
    eval_freq = EVAL_STEPS
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
        # 梯度裁剪
        if self.steps > self.warmup_steps:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        # 梯度更新
        self.optimizer.step()

        return loss_value.item()

    def _annealing_initial(self, total_train_steps):
        self.total_train_steps = total_train_steps
        # 学习率峰值
        self.peak_lr = self.optimizer.param_groups[0]["lr"]
        # 初始学习率
        self.initial_lr = 1e-8
        # 最小学习率
        self.min_lr = 0.1 * self.peak_lr

        # 学习率预热步数
        self.warmup_steps = int(self.config.warmup_steps_ratio * self.total_train_steps)
        # 学习率增长步长
        self.lr_increment = (self.peak_lr - self.initial_lr) / self.warmup_steps

    def _annealing(self):
        # 预热阶段
        if self.steps < self.warmup_steps:
            lr = self.initial_lr + self.steps * self.lr_increment
        else:
            # 非预热阶段的进度
            progress = (self.steps - self.warmup_steps) / (self.total_train_steps - self.warmup_steps)
            # 应用余弦衰减，计算学习率
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi*progress))

        # 在优化器上应用学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def _calc_train_loss_batch_amp(self, inputs, targets, scaler: GradScaler):
        self.optimizer.zero_grad()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=True
        ):
            outputs = self.model(inputs)
            loss_value = self.loss(outputs.transpose(-1, -2), targets)

        scaler.scale(loss_value).backward()

        # 梯度裁剪
        if self.steps > self.warmup_steps:
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        scaler.step(self.optimizer)
        scaler.update()


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


    def train(
            self,
            resume_model: bool = RESUME_MODEL,
            resume_optimizer: bool = RESUME_OPTIMIZER):
        if (self.config.models_dir/self.config.best_model_name).exists():
            try:
                tqdm.write(f"加载预训练模型...")
                checkpoint = torch.load(self.config.models_dir/self.config.best_model_name)
                if resume_model:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    tqdm.write(f"预训练模型加载成功...")
                if resume_optimizer:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.steps = checkpoint["steps"]
                    self.min_valid_loss = checkpoint["min_valid_loss"]
                    tqdm.write(f"优化器加载成功...")

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
        train_data_loader = create_data_loader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )

        valid_data_loader = create_data_loader(
            dataset=self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )


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
        result = Predictor.generate_top_k(
            model=self.model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_num_new_tokens=50,
            context_size=CONTEXT_LENGTH,
            top_k=50,
            temperature=0.95
        )
        print("Output text:\n", result)

        # 混合精度训练
        scaler = GradScaler(device=self.device.type, enabled=True)

        # 计算训练总步数
        total_train_steps = self.config.max_steps
        print(f"本次训练总步数为：{total_train_steps}")

        # 学习率退火参数初始化
        self._annealing_initial(total_train_steps)

        for epoch in range(self.config.epochs):
            for inputs, targets in tqdm(train_data_loader, desc="train"):
                # 记录步数
                if self.steps > self.config.max_steps:
                    return
                self.steps += 1
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # 学习率退火函数调用
                self._annealing()

                # 核心训练函数调用
                train_loss_batch = self._calc_train_loss_batch_amp(inputs, targets, scaler)

                # 累加本轮损失值
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
                        result = Predictor.generate_top_k(
                            model=self.model,
                            tokenizer=tokenizer,
                            input_ids=input_ids,
                            max_num_new_tokens=50,
                            context_size=CONTEXT_LENGTH,
                            top_k=50,
                            temperature=0.95
                        )
                        print("Output text:\n", result)

                    train_loss_accumulate = 0
                    count = 0



if __name__ == "__main__":
    torch.manual_seed(42)
    from src.lora.lora import create_lora_model
    # model_t = Qwen3Model.from_config()
    base_model = Qwen3Model.from_pretrained(BASE_MODEL_DIR / BASE_MODEL_NAME)
    lora_model = create_lora_model(base_model, 16, 16)

    trainer = Trainer.from_config(lora_model, train_config)

    trainer.train()
