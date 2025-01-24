
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

class RLTrainer:
    def __init__(
        self,
        model,
        reward_model,
        device: str = "cuda",
        learning_rate: float = 2e-5,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.reward_model = reward_model
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
    def compute_rewards(self, generated_outputs: List[str]) -> torch.Tensor:
        # 使用奖励模型计算奖励值
        with torch.no_grad():
            rewards = self.reward_model(generated_outputs)
        return rewards
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        self.model.train()
        
        # 生成序列
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=64,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # 计算奖励
        generated_texts = self.model.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        rewards = self.compute_rewards(generated_texts)
        
        # 计算PPO损失
        old_logprobs = self._compute_logprobs(outputs.scores, outputs.sequences)
        
        # 使用当前策略重新评估动作
        current_outputs = self.model(
            input_ids=outputs.sequences,
            attention_mask=torch.ones_like(outputs.sequences)
        )
        current_logprobs = self._compute_logprobs(
            current_outputs.logits, outputs.sequences
        )
        
        # 计算PPO比率和裁剪后的目标
        ratio = torch.exp(current_logprobs - old_logprobs)
        clip_param = 0.2
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * rewards
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
        }
    
    def train(
        self,
        train_dataloader,
        num_epochs: int,
        log_interval: int = 100,
    ):
        for epoch in range(num_epochs):
            epoch_stats = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                stats = self.train_step(batch)
                epoch_stats.append(stats)
                
                if step % log_interval == 0:
                    avg_stats = {
                        k: np.mean([s[k] for s in epoch_stats[-log_interval:]])
                        for k in stats.keys()
                    }
                    progress_bar.set_postfix(avg_stats)
    
    def _compute_logprobs(self, logits, sequences):
        # 计算序列的对数概率
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = sequences[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_logprobs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return token_logprobs.sum(dim=-1)

