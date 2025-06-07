import numpy as np
import pandas as pd
from typing import List, Tuple

class MotionDataGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_sequence(self, x0: float, v: float, n_points: int = 10) -> List[float]:
        """生成单个匀速运动序列
        Args:
            x0: 初始位置
            v: 速度
            n_points: 总点数（前5个 + 后5个）
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
            "Please follow the exact format of the example below to analyze the coordinates and make predictions.\n"
            "Your answer must contain [Analysis], [ANSWER], and [End] tags, with the final 5 predicted coordinates between [ANSWER] and [End] tags.\n\n"
            "Example:\n"
            "Input: (2.0), (3.5), (5.0), (6.5), (8.0)\n\n"
            "[Analysis]\n"
            "S1: Checking motion pattern\n"
            "   Δx1 = 3.5 - 2.0 = 1.5\n"
            "   Δx2 = 5.0 - 3.5 = 1.5\n"
            "   Δx3 = 6.5 - 5.0 = 1.5\n"
            "   Δx4 = 8.0 - 6.5 = 1.5\n"
            "   All intervals are approximately equal ≈ 1.5\n\n"
            "S2: Speed = Average(Δx/Δt) = 1.5 units/step\n\n"
            "S3: Prediction using average speed\n"
            "   Starting from x₀ = 8.0\n"
            "   x = x₀ + v·t = 8.0 + 1.5t\n\n"
            "[ANSWER]\n"
            "(9.5), (11.0), (12.5), (14.0), (15.5)\n"
            "[End]\n\n"
            
            f"Now analyze: {', '.join([f'({x:.1f})' for x in input_coords])}\n"
        )
        
        target = f"{', '.join([f'({x:.1f})' for x in target_coords])}"
        
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
                "query": prompt,
                "response": target,
            })
            
        return dataset