from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List

class MotionDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # GRPO需要的格式：必须包含 prompt，其他字段可选
        return {
            "prompt": item["query"],           # 输入提示
            "target": item["response"],        # 目标输出（用于计算奖励）
            "reasoning": item.get("reasoning", "")  # 推理过程（用于计算奖励）
        }