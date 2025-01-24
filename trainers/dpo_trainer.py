
from typing import Dict, Optional
import torch
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset

class O1DPOTrainer:
    def __init__(self, model, config, dataset):
        self.model = model
        self.config = config
        self.dataset = dataset
        
    def prepare_dpo_dataset(self, dataset: Dataset) -> Dataset:
        """准备DPO训练所需的数据集格式"""
        def format_dpo_data(examples):
            return {
                "prompt": examples["prompt"],
                "chosen": examples["chosen_response"],
                "rejected": examples["rejected_response"]
            }
            
        return dataset.map(format_dpo_data, remove_columns=dataset.column_names)
        
    def train(self, output_dir: str):
        """执行DPO训练"""
        # 准备训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            fp16=True,
            report_to="wandb"
        )
        
        # 初始化DPO训练器
        dpo_trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            beta=self.config.dpo_beta,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"] if "validation" in self.dataset else None,
            tokenizer=self.model.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length
        )
        
        # 开始训练
        dpo_trainer.train()
        
        # 保存模型
        dpo_trainer.save_model()
        self.model.tokenizer.save_pretrained(output_dir)

