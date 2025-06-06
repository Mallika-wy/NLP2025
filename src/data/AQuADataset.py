"""
数据加载器实现
"""
import json
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
from .data_loader import preprocess_aqua_question, get_correct_answer_letter

class AQuADataset(Dataset):
    """AQuA数据集加载器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据集
        
        Args:
            dataset_path: 数据集文件路径
        """
        self.data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含问题信息的字典
        """
        item = self.data[idx]
        return {
            'questions': preprocess_aqua_question(item),
            'options': item['options'],
            'corrects': get_correct_answer_letter(item)
        }

def create_dataloader(
    dataset_path: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset_path: 数据集文件路径
        batch_size: 批处理大小
        num_workers: 工作进程数
        shuffle: 是否打乱数据
        
    Returns:
        数据加载器实例
    """
    dataset = AQuADataset(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )