
from typing import List, Dict, Any
import torch
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

class O1Evaluator:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_reasoning(
        self,
        eval_dataset,
        metrics: List[str] = ["accuracy", "reasoning_quality"],
        num_samples: int = None
    ) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {}
        
        if num_samples:
            eval_dataset = eval_dataset.select(range(num_samples))
        
        results = []
        for item in tqdm(eval_dataset, desc="Evaluating"):
            with torch.no_grad():
                output = self.model.generate_with_reasoning(item["question"])
                
                result = {
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "predicted_answer": output["final_answer"],
                    "reasoning_steps": output["reasoning_steps"]
                }
                results.append(result)
        
        # 计算评估指标
        if "accuracy" in metrics:
            all_metrics["accuracy"] = self._compute_accuracy(results)
        if "reasoning_quality" in metrics:
            all_metrics["reasoning_quality"] = self._evaluate_reasoning_quality(results)
            
        return all_metrics
    
    def _compute_accuracy(self, results: List[Dict]) -> float:
        correct = 0
        total = len(results)
        
        for result in results:
            if self._normalize_answer(result["predicted_answer"]) == \
               self._normalize_answer(result["ground_truth"]):
                correct += 1
                
        return correct / total if total > 0 else 0.0
    
    def _evaluate_reasoning_quality(self, results: List[Dict]) -> Dict[str, float]:
        """评估推理质量的多个方面"""
        total_steps = []
        coherence_scores = []
        
        for result in results:
            # 计算推理步骤数
            num_steps = len(result["reasoning_steps"])
            total_steps.append(num_steps)
            
            # 评估推理连贯性（简单示例）
            coherence = self._compute_coherence(result["reasoning_steps"])
            coherence_scores.append(coherence)
        
        return {
            "avg_reasoning_steps": np.mean(total_steps),
            "coherence_score": np.mean(coherence_scores)
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """标准化答案格式以进行比较"""
        return answer.lower().strip()
    
    def _compute_coherence(self, reasoning_steps: List[str]) -> float:
        """
        计算推理步骤的连贯性分数
        这是一个简化的实现，实际应用中可能需要更复杂的评估方法
        """
        if not reasoning_steps:
            return 0.0
            
        # 简单地检查步骤之间是否有逻辑连接词
        coherence_markers = ["therefore", "because", "so", "thus", "as a result"]
        score = 0
        
        for step in reasoning_steps:
            if any(marker in step.lower() for marker in coherence_markers):
                score += 1
                
        return score / len(reasoning_steps) if reasoning_steps else 0.0
    
    def save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """将评估结果保存到文件"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

