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
Alibaba's `Macro-O1`:
```bibtxt
@misc{zhao2024marcoo1openreasoningmodels,
      title={Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions}, 
      author={Yu Zhao and Huifeng Yin and Bo Zeng and Hao Wang and Tianqi Shi and Chenyang Lyu and Longyue Wang and Weihua Luo and Kaifu Zhang},
      year={2024},
      eprint={2411.14405},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.14405}, 
}
```

---
- `LLaMA-O1`: [llama-o1](https://github.com/SimpleBerry/LLaMA-O1)

---

`Open-O1`:
```bibtxt
@article{
    supplement,
    title={},
    author={OpenO1 Team},
    journal={},
    url={},
    year={}
}
```


---

## Useful Blogs
- [OpenAI o1 复现——过程奖励模型（PRM）](https://zhuanlan.zhihu.com/p/1246547399)
- [上交大O1复现](https://zhuanlan.zhihu.com/p/10180139030)
- [OpenAI o1 self-play RL 技术路线推演](https://zhuanlan.zhihu.com/p/720106482)
- [Reverse-o1:OpenAI o1原理逆向工程图解](https://zhuanlan.zhihu.com/p/721952915)