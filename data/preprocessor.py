
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any

class O1Dataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 2048
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # 添加思维链提示
        enhanced_text = f"Let's approach this step by step:\n{text}"
        
        # tokenization
        encodings = self.tokenizer(
            enhanced_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 准备输入
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # 创建标签（用于自回归训练）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_dataset(
    raw_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int = 2048
) -> O1Dataset:
    """
    准备训练数据集
    
    Args:
        raw_data: 原始数据列表
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        处理后的数据集
    """
    # 提取文本
    texts = [item["text"] for item in raw_data]
    
    # 创建数据集
    dataset = O1Dataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return dataset

