"""
数据加载和预处理工具
"""
import json
import os
import random
from typing import List, Dict, Any
import configs.config as config

def load_aqua_dataset():
    """加载AQuA数据集.
    
    数据集格式示例 (AQuA-test.json):
    {
        "question": "What is the value of x in the equation 2x + 3 = 7?",
        "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 5"],
        "rationale": "2x + 3 = 7 -> 2x = 4 -> x = 2. So the answer is B.",
        "correct": "B"
    }
    """
    if not os.path.exists(config.data.dataset_path):
        print(f"Error: Dataset file not found at {config.data.dataset_path}")
        return []

    data = []
    try:
        with open(config.data.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error reading or parsing dataset file: {e}")
        return []
    
    if config.data.max_samples is not None and config.data.max_samples < len(data):
        random.seed(config.experiment.random_seed) # 保证可复现性
        data = random.sample(data, config.data.max_samples)
        print(f"Loaded {len(data)} samples from {config.data.dataset_path} (randomly sampled).")
    else:
        print(f"Loaded {len(data)} samples from {config.data.dataset_path}.")
    return data

def preprocess_aqua_question(item):
    """预处理单个AQuA问题，将其转换为适合模型的格式。"""
    question = item.get("question", "")
    options = item.get("options", [])
    # 将选项格式化到问题中
    formatted_options = "\n".join(options)
    full_question = f"{question}\nOptions:\n{formatted_options}"
    return full_question

def get_correct_answer_letter(item):
    """获取正确答案的字母选项"""
    return item.get("correct", "").upper()

# 测试 (可选)
if __name__ == '__main__':
    # 需要在项目根目录下创建一个 data/AQuA-test.json 文件用于测试
    # 示例 AQuA-test.json 内容 (一行一个json对象):
    # {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "options": ["A) 24", "B) 72", "C) 48", "D) 144", "E) 96"], "rationale": "Natalia sold 48 clips in April. She sold half as many clips in May which is 48/2 = 24. So she sold 48 + 24 = 72 clips altogether. The answer is B.", "correct": "B"}
    # {"question": "What is the value of x in the equation 2x + 3 = 7?", "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 5"], "rationale": "2x + 3 = 7 -> 2x = 4 -> x = 2. So the answer is B.", "correct": "B"}
    
    # 为了能够运行测试，请确保 config.py 中的 DATASET_PATH 指向一个有效的 AQuA JSON Lines 文件
    # 且该文件位于如 self_consistency_experiment/data/AQuA-test.json
    # 如果没有这个文件，下面的测试会因找不到文件而跳过
    
    if not os.path.exists(config.DATASET_PATH):
        print(f"Skipping test: Dataset file {config.DATASET_PATH} not found.")
        print("Please create a dummy AQuA-test.json in the 'data' directory to run this test.")
    else:
        dataset = load_aqua_dataset()
        if dataset:
            sample_item = dataset[0]
            print("\nSample item:")
            print(json.dumps(sample_item, indent=2))
            
            print("\nPreprocessed question:")
            processed_q = preprocess_aqua_question(sample_item)
            print(processed_q)
            
            print("\nCorrect answer letter:")
            correct_letter = get_correct_answer_letter(sample_item)
            print(correct_letter)
        else:
            print("No data loaded, skipping further tests.") 