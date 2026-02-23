import sys
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.configuration.config import *

import torch
import time
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler
from dataclasses import dataclass
from tqdm import tqdm


from src.model.qwen3_model import Qwen3Model
from src.preprocess.dpo_dataset import DPODataset, create_dpo_data_loader
from src.preprocess.tokenizer import Tokenizer
from src.preprocess.instruct_processor import format_input
from src.predict.predictor import Predictor


@dataclass
class DPOTrainConfig:
    policy_model_file: Path = MODELS_DIR/LORA_MODEL_NAME
    refer_model_file: Path = MODELS_DIR/LORA_MODEL_NAME
    models_output_dir: Path = MODELS_DIR
    logs_dir: Path =  LOGS_DIR/"dpo"
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    max_steps = MAX_STEPS
    learning_rate = LEARNING_RATE
    warmup_steps_ratio = WARMUP_STEPS_RATIO
    weight_decay = WEIGHT_DECAY
    eval_freq = EVAL_STEPS
    dpo_models_name = DPO_BEST_MODEL_WITH_LORA
    beta = DPO_BETA



train_config = DPOTrainConfig()
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
            self,
            cfg: DPOTrainConfig,
            policy_model: Qwen3Model,
            refer_model: Qwen3Model,
            tokenizer: Tokenizer,
            optimizer: optim.Optimizer,
            loss: nn.CrossEntropyLoss,
            train_dataset: DPODataset,
            valid_dataset: DPODataset,
            device: torch.device
    ):
        self.config = cfg
        self.policy_model = policy_model
        self.refer_model = refer_model
        self.tokenizer = tokenizer
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
            policy_model: Qwen3Model,
            refer_model: Qwen3Model,
            train_cfg: DPOTrainConfig = train_config,
            train_dataset_file = str(DPO_PROCESSED_DIR/TRAIN_DATA_FILE),
            valid_dataset_file = str(DPO_PROCESSED_DIR/VALID_DATA_FILE),
            device: torch.device = default_device
    ):
        # 模型加载
        # 策略模型
        policy_model = policy_model.to(device)
        # 参考模型, 禁止参数更新
        refer_model = refer_model.eval().to(device)
        print("冻结参考模型所有参数...")
        for param in refer_model.parameters():
            param.requires_grad = False

        total_params_in_refer_model = sum(p.numel() for p in refer_model.parameters() if p.requires_grad)
        print(f"检查参考模型的可更新参数量为：{total_params_in_refer_model}")

        assert total_params_in_refer_model == 0, "参考模型的参数未冻结"

        # 分词器
        tokenizer = Tokenizer.from_vocab(special_char_set=SPECIAL_CHAR_SET)

        # 优化器，AdamW
        optimizer = optim.AdamW(
            policy_model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )

        # 损失函数
        loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        # 加载数据集
        train_dataset = DPODataset.from_json(train_dataset_file)
        valid_dataset = DPODataset.from_json(valid_dataset_file)

        # print(model)

        return cls(
            cfg = train_cfg,
            policy_model=policy_model,
            refer_model=refer_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss = loss,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            device=device
        )


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

    @classmethod
    def _compute_log_probs(cls, logits, labels, selection_mask=None):
        """
        计算对数概率函数
        :param logits: 经过model后的词表中词元的概率分布
        :param labels: 标签，取model的输入
        :param selection_mask: 屏蔽提示词，不屏蔽答案的掩码
        :return:整组tokens对数概率的平均值
        """
        # 往前错一位，[batch_size, token_ids_list]->[batch_size, num_tokens, token_id]
        labels = labels[:, 1:].clone().unsqueeze(-1)
        # [batch_size, num_tokens, vocab_size]
        logits =  logits[:, :-1, :]
        # 先求词表中的概率分布求对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        # 按照labels对应词元序号，把对应的对数概率从词表中扒出来
        # [batch_size, num_tokens, vocab_size]->[batch_size, log_probs_list]
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels
        ).squeeze(-1)

        # 如果要打掩码的话
        if selection_mask is not None:
            mask = selection_mask[:, :-1].clone()
            masked_log_probs_sum = (selected_log_probs * mask).sum(dim=-1)
            # 对答案部分的对数概率求平均值
            mask_sum = mask.sum(dim=-1).clamp(min=1)
            avg_log_probs = masked_log_probs_sum / mask_sum
            return avg_log_probs
        else:
            return selected_log_probs.mean(dim=-1)

    def _compute_dpo_loss(
            self,
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs
    ):
        # 按照公式：策略模型的输出比上参考模型的输出
        chosen_log_probs_ratios = policy_chosen_log_probs - reference_chosen_log_probs
        rejected_log_probs_ratios = policy_rejected_log_probs - reference_rejected_log_probs
        logits = chosen_log_probs_ratios - rejected_log_probs_ratios
        # 代入公式计算损失
        losses = -F.logsigmoid(self.config.beta * logits)
        # 分离出计算图
        chosen_rewards = chosen_log_probs_ratios.detach().clone()
        rejected_rewards = rejected_log_probs_ratios.detach().clone()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def _compute_dpo_loss_batch(
            self,
            chosen,
            chosen_mask,
            rejected,
            rejected_mask
    ):
        policy_chosen_log_probs = self._compute_log_probs(
            logits=self.policy_model(chosen),
            labels=chosen,
            selection_mask=chosen_mask
        )

        policy_rejected_log_probs = self._compute_log_probs(
            logits=self.policy_model(rejected),
            labels=rejected,
            selection_mask=rejected_mask
        )

        reference_chosen_log_probs = self._compute_log_probs(
            logits=self.refer_model(chosen),
            labels=chosen,
            selection_mask=chosen_mask
        )

        reference_rejected_log_probs = self._compute_log_probs(
            logits=self.refer_model(rejected),
            labels=rejected,
            selection_mask=rejected_mask
        )

        loss, chosen_rewards, rejected_rewards = self._compute_dpo_loss(
            policy_chosen_log_probs=policy_chosen_log_probs,
            policy_rejected_log_probs=policy_rejected_log_probs,
            reference_chosen_log_probs=reference_chosen_log_probs,
            reference_rejected_log_probs=reference_rejected_log_probs
        )

        return loss, chosen_rewards, rejected_rewards


    def _calc_train_loss_batch_amp(
            self,
            chosen,
            chosen_mask,
            rejected,
            rejected_mask,
            scaler: GradScaler
    ):
        self.optimizer.zero_grad()

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=True
        ):
            loss_value, chosen_rewards, rejected_rewards = self._compute_dpo_loss_batch(
                chosen=chosen,
                chosen_mask=chosen_mask,
                rejected=rejected,
                rejected_mask=rejected_mask
            )

        scaler.scale(loss_value).backward()

        # 梯度裁剪
        if self.steps > self.warmup_steps:
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)

        scaler.step(self.optimizer)
        scaler.update()


        return loss_value.item(), chosen_rewards.item(), rejected_rewards.item()

    # 计算验证损失
    def _calc_valid_loss(self, valid_data_loader):
        self.policy_model.eval()
        self.refer_model.eval()
        with torch.no_grad():
            valid_loss_accumulate = 0
            chosen_rewards_accumulate = 0
            rejected_rewards_accumulate = 0
            count = 0
            for chosen, chosen_mask, rejected, rejected_mask in valid_data_loader:
                chosen = chosen.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                rejected = rejected.to(self.device)
                rejected_mask = rejected_mask.to(self.device)
                loss_value, chosen_rewards, rejected_rewards = self._compute_dpo_loss_batch(
                    chosen=chosen,
                    chosen_mask=chosen_mask,
                    rejected=rejected,
                    rejected_mask=rejected_mask
                )
                valid_loss_accumulate += loss_value * chosen.shape[0]
                chosen_rewards_accumulate += chosen_rewards * chosen.shape[0]
                rejected_rewards_accumulate += rejected_rewards * rejected.shape[0]
                count += chosen.shape[0]

            return valid_loss_accumulate / count, chosen_rewards_accumulate / count, rejected_rewards_accumulate / count



    def train(self):
        print("倒计时30秒")
        for _ in tqdm(range(30), desc="启动前等待", bar_format="{desc}: {remaining}s"):
            time.sleep(1)
        print("训练开始...")

        self.policy_model.train()
        # 准备数据加载器
        train_data_loader = create_dpo_data_loader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            pad_token_id=self.tokenizer.eos_id
        )

        valid_data_loader = create_dpo_data_loader(
            dataset=self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            pad_token_id=self.tokenizer.eos_id
        )

        train_loss_acc = 0.0
        train_chosen_rewards_acc = 0.0
        train_rejected_rewards_acc = 0.0
        count = 0
        writer = SummaryWriter(
            log_dir=str(self.config.logs_dir / time.strftime("%Y-%m-%d_%H-%M-%S"))
        )

        # 测试用例

        entry = {"question": "你是谁？", "answer": ""}
        text = format_input(entry)
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(default_device)
        # 初始文本生成测试
        result = Predictor.generate_top_k(
            model=self.policy_model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            max_num_new_tokens=50,
            context_size=CONTEXT_LENGTH,
            top_k=50,
            temperature=0.3
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
            for chosen, chosen_mask, rejected, rejected_mask in tqdm(train_data_loader, desc="train"):
                # 记录步数
                if self.steps > self.config.max_steps:
                    return
                self.steps += 1
                chosen = chosen.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                rejected = rejected.to(self.device)
                rejected_mask = rejected_mask.to(self.device)

                # 学习率退火函数调用
                self._annealing()

                # 核心训练函数调用
                train_loss_batch, train_chosen_rewards, train_rejected_rewards = self._calc_train_loss_batch_amp(
                    chosen=chosen,
                    chosen_mask=chosen_mask,
                    rejected=rejected,
                    rejected_mask=rejected_mask,
                    scaler=scaler
                )

                # 累加本轮损失值
                train_loss_acc += train_loss_batch * chosen.shape[0]
                train_chosen_rewards_acc += train_chosen_rewards * chosen.shape[0]
                train_rejected_rewards_acc += train_rejected_rewards * rejected.shape[0]
                count += chosen.shape[0]
                if self.steps % self.config.eval_freq == 0:
                    train_loss = train_loss_acc / count
                    train_chosen_reward = train_chosen_rewards_acc / count
                    train_rejected_reward = train_rejected_rewards_acc / count
                    print("computing validation loss, please wait a minute~")

                    valid_loss, valid_chosen_reward, valid_rejected_reward = self._calc_valid_loss(valid_data_loader)
                    tqdm.write(f"\nEPOCH {epoch + 1}|STEP {self.steps}"
                               f"|train_loss:{train_loss:.6f}|valid_loss: {valid_loss:.6f}"
                               f"|train_reward_margins:{train_chosen_reward - train_rejected_reward:.6f}"
                               f"|valid_reward_margins:{valid_chosen_reward - valid_rejected_reward:.6f}")

                    writer.add_scalar("train_loss", train_loss, self.steps)
                    writer.add_scalar("valid_loss", valid_loss, self.steps)
                    writer.add_scalar("train_reward_margins", train_chosen_reward - train_rejected_reward, self.steps)
                    writer.add_scalar("valid_reward_margins", valid_chosen_reward - valid_rejected_reward, self.steps)


                    if valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        # 保存模型参数
                        torch.save({
                            "model_state_dict": self.policy_model.state_dict()
                        },
                            str(self.config.models_output_dir/self.config.dpo_models_name)
                        )
                        print("the newest model have been saved.")

                        # 文本生成测试
                        result = Predictor.generate_top_k(
                            model=self.policy_model,
                            tokenizer=self.tokenizer,
                            input_ids=input_ids,
                            max_num_new_tokens=50,
                            context_size=CONTEXT_LENGTH,
                            top_k=50,
                            temperature=0.3
                        )
                        print("Output text:\n", result)

                    train_loss_acc = 0
                    train_chosen_rewards_acc = 0
                    train_rejected_rewards_acc = 0
                    count = 0



if __name__ == "__main__":
    torch.manual_seed(123)
    from src.model.qwen3_model import Qwen3Model
    from src.lora.lora import create_lora_model
    from src.configuration.config import *

    # 独立创建两个model
    policy_model_t = Qwen3Model.from_pretrained(LORA_MODELS_DIR/MERGED_LORA_MODEL_NAME)
    policy_model_t = create_lora_model(policy_model_t, rank=RANK, alpha=ALPHA)
    refer_model_t = Qwen3Model.from_pretrained(LORA_MODELS_DIR/MERGED_LORA_MODEL_NAME)

    trainer = Trainer.from_config(policy_model_t, refer_model_t)

    trainer.train()
