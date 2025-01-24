
import torch
import torch.nn as nn
from transformers import AutoModel

from typing import Optional, List, Dict, Any

class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用与基础模型相同的架构
        self.backbone = AutoModel.from_pretrained(
            config.base_model_name,
            trust_remote_code=True
        )
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1)
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用最后一个token的隐藏状态预测奖励
        last_hidden = outputs.last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden)
        
        return rewards

