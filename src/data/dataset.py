from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List

class MotionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 设置特殊token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取输入和目标
        prompt = item["query"]
        response = item["response"]
        
        # tokenize输入
        encoded = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512  # 根据需要调整
        )
        
        # 去除batch维度
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "prompt": prompt,
            "response": response,
            "reasoning": item.get("reasoning", "")
        }