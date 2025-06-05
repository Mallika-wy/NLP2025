# NLP Final Project : 基于大语言模型的思维链推理

## 任务要求

在Lab4基础上，基于chatgpt或者开源的LLAMA、vicuna等LLM，进行一些开放性的进阶探索
1. 引入多条推理路径和Self-Consistency
Self-Consistency Improves Chain of Thought Reasoning in Language Models
2. 针对不同推理路径构建verifier：：
Making Large Language Models Better Reasoners with Step-Aware Verifier
3. 与外部知识结合，利用检索增强技术赋能COT：
Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions


数据集：AQuA: https://github.com/deepmind/AQuA


## 项目结构

```
SelfConsistency/
├── src/                      # 源代码目录
│   ├── core/                 # 核心功能实现
│   ├── utils/               # 工具函数
│   └── data/                # 数据处理相关
├── configs/                 # 配置文件目录
├── data/                   # 数据存储目录
├── results/                # 结果输出目录
├── tests/                  # 测试代码目录
└── main.py                # 主入口文件
```

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- Accelerate

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd SelfConsistency
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据：
- 下载 AQuA 数据集
- 将数据文件放在 `data/` 目录下

## 配置

在 `configs/config.py` 中设置：
- 模型路径
- 数据集路径
- 实验参数（推理路径数量、温度等）

## 运行

```bash
python main.py
```

## 结果

实验结果将保存在 `results/` 目录下：
- experiment_log.txt：实验日志
- results.json：详细结果和评估指标


