import numpy as np
import pandas as pd
from typing import List, Tuple

class MotionDataGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_sequence(self, x0: float, v: float, n_points: int = 13) -> List[float]:
        """生成单个匀速运动序列
        Args:
            x0: 初始位置
            v: 速度
            n_points: 总点数（前5个 + 后8个）
        Returns:
            完整的坐标序列
        """
        t = np.arange(n_points)
        x = x0 + v * t
        return x.tolist()
    
    def format_coordinates(self, coords: List[float]) -> Tuple[str, str]:
        """将坐标格式化为输入和目标字符串
        Args:
            coords: 完整的坐标序列
        Returns:
            (input_text, target_text): 输入提示和目标输出
        """
        input_coords = coords[:5]
        target_coords = coords[5:]
        
        prompt = (
            "给定以下一维匀速运动的坐标序列：\n"
            f"{', '.join([f'({x:.2f})' for x in input_coords])}\n"
            "请分析这些点之间的关系，并预测接下来的8个坐标点。\n\n"
            "思考步骤：\n"
            "1. 计算相邻点之间的差值\n"
            "2. 确认运动是否为匀速运动\n"
            "3. 计算速度\n"
            "4. 预测后续坐标\n\n"
            "预测结果："
        )
        
        target = f"{', '.join([f'({x:.2f})' for x in target_coords])}"
        
        return prompt, target
    
    def generate_dataset(self, split='train', n_samples=1000):
        """生成数据集
        Args:
            split: 数据集类型 ('train', 'test_in', 'test_out')
            n_samples: 样本数量
        Returns:
            数据集字典列表
        """
        if split == 'train' or split == 'test_in':
            x0_range = (-5.0, 5.0)
            v_range = (0.5, 2.0)
        else:  # test_out
            x0_range = (-5.0, 5.0)
            v_range = (2.0, 3.0)
            
        dataset = []
        for _ in range(n_samples):
            x0 = np.random.uniform(*x0_range)
            v = np.random.uniform(*v_range)
            
            coords = self.generate_sequence(x0, v)
            prompt, target = self.format_coordinates(coords)
            
            dataset.append({
                "input_ids": prompt,
                "query": prompt,
                "response": target,
            })
            
        return dataset