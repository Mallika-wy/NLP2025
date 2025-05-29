from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List

class MotionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer_name: str):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 编码输入文本
        inputs = self.tokenizer(
            item["query"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 编码目标文本
        response = self.tokenizer(
            item["response"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "response": item["response"],
            "response_ids": response["input_ids"].squeeze()
        }