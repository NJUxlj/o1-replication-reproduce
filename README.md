# o1-replication-reproduce
---
Use Qwen2.5-3B as the base model to reproduce the technical route of OpenAI-O1.



## Details
- we largely use `Claude-3.5-sonnet` and RAG framework to search the code and the detials of route of the OpenAI-O1.



## Characteristics of O1
- 使用大规模强化学习训练，特别强调推理能力
- 采用Chain of Thought (CoT)方法来提升推理表现
- 模型在回答前会进行更深入的思考
- 在代码和问题解决方面有特殊优化



## Our Project Structure
```Plain Txt

o1-implementation/
├── config/
│   └── model_config.py
├── data/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   └── datasets/
│       └── __init__.py
├── models/
│   ├── __init__.py
│   ├── o1_model.py
│   └── reward_model.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── rl_trainer.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
├── utils/
│   ├── __init__.py
│   └── logging_utils.py
├── requirements.txt
└── main.py


```


## Run
```bash
python train.py
```



## TODO
- [ ] 1. 复现OpenAI-O1的技术路线
- [ ] 2. 复现OpenAI-O1的代码
- [ ] 3. 复现OpenAI-O1的效果






## Citation
```bibtxt

```

```bibtxt

```

```bibtxt

```
