
from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_model_name: str = "Qwen/Qwen2.5-3B"
    max_sequence_length: int = 2048
    hidden_size: int = 1024
    num_attention_heads: int = 32
    intermediate_size: int = 4096
    num_hidden_layers: int = 32
    attention_probs_dropout_prob: int = 0.1
    hidden_dropout_prob: int = 0.1
    
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
@dataclass
class RLConfig:
    ppo_epochs: int = 4
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    kl_penalty_coef: float = 0.1

