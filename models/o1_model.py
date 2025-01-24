
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

class ReasoningModule(nn.Module):
    """
    ReasoningModule（推理模块）

        实现了系统思维(System of Thought)和思维链(Chain of Thought)
        使用多头注意力机制进行深度推理
        包含门控机制来融合原始表示和推理结果
    
    """
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.thought_projection = nn.Linear(hidden_size, hidden_size)
        self.reasoning_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.thought_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.thought_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 投影到思维空间
        thought_states = self.thought_projection(hidden_states)
        
        # 多头注意力用于推理
        reasoned_states, _ = self.reasoning_attention(
            thought_states, 
            thought_states, 
            thought_states, 
            key_padding_mask=attention_mask
        )
        
        # 门控机制融合原始表示和推理结果
        gate = torch.sigmoid(self.thought_gate(
            torch.cat([hidden_states, reasoned_states], dim=-1)
        ))
        gated_output = gate * reasoned_states + (1 - gate) * hidden_states
        
        # 归一化
        output = self.thought_norm(gated_output)
        return output

class DeliberationModule(nn.Module):
    """
    深度思考模块：实现多步推理和决策
            实现多步推理过程
            每一步都会产生置信度得分
            使用残差连接确保信息的有效传递
    
    """
    def __init__(self, hidden_size: int, num_steps: int = 3):
        super().__init__()
        self.num_steps = num_steps
        self.step_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_steps)
        ])
        self.confidence_scorer = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_state = hidden_states
        confidences = []
        
        for step_net in self.step_nets:
            # 每一步深入思考
            current_state = current_state + step_net(current_state)
            # 计算当前步骤的置信度
            confidence = torch.sigmoid(self.confidence_scorer(current_state))
            confidences.append(confidence)
            
        # 返回最终状态和各步骤置信度
        return current_state, torch.stack(confidences, dim=1)

class O1Model(nn.Module):
    """
    O1模型主体
        基于Qwen2.5-3B作为底层模型
        集成了推理模块和深度思考模块
        包含用于强化学习的价值头
        实现了高效的生成方法
        支持强化学习训练
    
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen1.5-3B",
        device: str = "cuda",
        load_in_8bit: bool = False
    ):
        super().__init__()
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=load_in_8bit
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
        
        hidden_size = self.base_model.config.hidden_size
        
        # 初始化各个模块
        self.reasoning_module = ReasoningModule(hidden_size)
        self.deliberation_module = DeliberationModule(hidden_size)
        
        # 输出头
        self.output_projection = nn.Linear(hidden_size, self.base_model.config.vocab_size)
        
        # 用于强化学习的价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        
        # 获取基础模型的隐藏状态
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = base_outputs.hidden_states[-1]
        
        # 应用推理模块
        reasoned_states = self.reasoning_module(
            hidden_states,
            attention_mask
        )
        
        # 应用深度思考模块
        deliberated_states, confidence_scores = self.deliberation_module(reasoned_states)
        
        # 计算logits
        logits = self.output_projection(deliberated_states)
        
        # 计算价值估计（用于RL）
        value_estimates = self.value_head(deliberated_states).squeeze(-1)
        
        outputs = {
            "logits": logits,
            "value_estimates": value_estimates,
            "confidence_scores": confidence_scores,
            "hidden_states": deliberated_states
        }
        
        if labels is not None:
            # 计算语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs["loss"] = lm_loss
            
        return outputs if return_dict else (logits, value_estimates)
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        min_length: int = 0,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> torch.LongTensor:
        """生成文本的方法"""
        
        # 初始化生成参数
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        device = input_ids.device
        
        generated = input_ids
        
        # 自回归生成
        for _ in range(max_length - current_length):
            # 获取模型输出
            outputs = self.forward(generated, return_dict=True)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用temperature
            next_token_logits = next_token_logits / temperature
            
            # 应用top_k过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 应用top_p过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 拼接生成的token
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # 检查是否生成了终止符
            if (next_tokens == self.tokenizer.eos_token_id).any():
                break
        
        return generated
    
    def prepare_inputs_for_rl(
        self,
        prompt_ids: torch.LongTensor,
        response_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """准备强化学习的输入"""
        
        # 合并prompt和response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            response_mask = torch.ones_like(response_ids)
            attention_mask = torch.cat([attention_mask, response_mask], dim=1)
            
        # 创建response mask用于计算response的reward
        response_mask = torch.zeros_like(input_ids)
        response_mask[:, prompt_ids.shape[1]:] = 1
        
        return input_ids, attention_mask, response_mask
    
    def compute_rewards(
        self,
        hidden_states: torch.Tensor,
        response_mask: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """计算强化学习的奖励"""
        
        # 基于置信度的奖励
        confidence_reward = confidence_scores.mean(dim=1)
        
        # 基于隐藏状态质量的奖励
        quality_scores = torch.norm(hidden_states, dim=-1)
        quality_reward = (quality_scores * response_mask).sum(dim=1) / response_mask.sum(dim=1)
        
        # 合并奖励
        combined_reward = 0.7 * confidence_reward + 0.3 * quality_reward
        
        return combined_reward
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        # 保存基础模型
        self.base_model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # 保存其他模块
        torch.save({
            'reasoning_module': self.reasoning_module.state_dict(),
            'deliberation_module': self.deliberation_module.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'value_head': self.value_head.state_dict()
        }, f"{save_directory}/o1_modules.pt")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """加载预训练模型"""
        model = cls(base_model_name=load_directory, **kwargs)
        
        # 加载其他模块
        modules_state = torch.load(f"{load_directory}/o1_modules.pt")
        model.reasoning_module.load_state_dict(modules_state['reasoning_module'])
        model.deliberation_module.load_state_dict(modules_state['deliberation_module'])
        model.output_projection.load_state_dict(modules_state['output_projection'])
        model.value_head.load_state_dict(modules_state['value_head'])
        
        return model

