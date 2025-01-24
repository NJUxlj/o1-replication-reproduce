
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm

class O1Trainer:
    def __init__(
        self,
        model,
        reward_model,
        train_dataloader,
        eval_dataloader,
        config
    ):
        self.model = model
        self.reward_model = reward_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # 优化器设置
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
        
        # 初始化wandb
        wandb.init(project="o1-implementation")
        
    def train_step(self, batch):
        self.model.train()
        
        # 前向传播
        outputs = self.model(**batch)
        loss = outputs["loss"]
        
        # 计算奖励并应用PPO更新
        with torch.no_grad():
            rewards = self.reward_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        
        # PPO损失计算
        old_logprobs = outputs["logits"].log_softmax(-1)
        new_logprobs = outputs["logits"].detach().log_softmax(-1)
        ratio = (new_logprobs - old_logprobs).exp()
        
        # 裁剪PPO目标
        clip_param = 0.2
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * rewards
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # 总损失
        total_loss = loss + self.config.ppo_loss_coef * ppo_loss
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            "loss": total_loss.item(),
            "ppo_loss": ppo_loss.item(),
            "rewards": rewards.mean().item()
        }
    
    def train(self):
        global_step = 0
        best_eval_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            for batch in tqdm(self.train_dataloader):
                metrics = self.train_step(batch)
                
                # 记录指标
                wandb.log(metrics)
                
                global_step += 1
                
                # 评估
                if global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    wandb.log(eval_metrics)
                    
                    # 保存最佳模型
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        self.save_model("best_model")
                
                if global_step >= self.config.max_steps:
                    break
                    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_rewards = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                rewards = self.reward_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                total_loss += loss.item()
                total_rewards += rewards.mean().item()
                num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_rewards": total_rewards / num_batches
        }
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        self.model.config.save_pretrained(path)

