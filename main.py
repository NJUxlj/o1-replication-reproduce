
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer

from config.model_config import ModelConfig, TrainingConfig, RLConfig
from models.o1_model import O1Model
from models.reward_model import RewardModel
from trainers.sft_trainer import O1Trainer
from data.preprocessor import prepare_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # 配置
    model_config = ModelConfig()
    training_config = TrainingConfig()