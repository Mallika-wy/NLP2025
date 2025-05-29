import numpy as np
import re
from typing import List, Tuple
import torch
from sklearn.metrics import mean_squared_error

class RewardFunction:
    def __init__(self, config):
        self.config = config
        # 配置各个奖励组件的权重
        self.weights = {
            'accuracy': 0.4,    # 坐标准确性权重
            'format': 0.1,      # 格式正确性权重
            'cot': 0.3,        # 推理过程权重
            'trend': 0.2       # 变化趋势权重
        }
        
    def parse_coordinates(self, coord_str: str) -> List[float]:
        """解析坐标字符串为数值列表
        Args:
            coord_str: 形如 "(1.23), (4.56), (7.89)" 的坐标字符串
        Returns:
            坐标值列表
        """
        try:
            # 使用正则表达式提取数字
            coords = re.findall(r'\(([-+]?\d*\.?\d+)\)', coord_str)
            return [float(x) for x in coords]
        except:
            return []

    def calculate_accuracy_reward(self, predicted: str, actual: str) -> float:
        """计算预测坐标的准确性奖励
        
        使用MSE的负值作为奖励，并进行归一化
        """
        pred_coords = self.parse_coordinates(predicted)
        actual_coords = self.parse_coordinates(actual)
        
        if not pred_coords or not actual_coords or len(pred_coords) != len(actual_coords):
            return -1.0
            
        # 计算MSE
        mse = mean_squared_error(actual_coords, pred_coords)
        # 将MSE转换为0到1之间的奖励值
        reward = np.exp(-mse)  # 使用指数函数将MSE转换为奖励
        return reward

    def calculate_format_reward(self, predicted: str) -> float:
        """计算输出格式的正确性奖励
        
        检查是否符合 "(x.xx)" 的格式要求
        """
        # 检查格式是否正确
        pattern = r'^\s*(\([-+]?\d*\.?\d+\)[\s,]*)+\s*$'
        if re.match(pattern, predicted):
            # 检查是否包含正确数量的坐标（8个）
            coords = re.findall(r'\(([-+]?\d*\.?\d+)\)', predicted)
            if len(coords) == 8:
                return 1.0
            else:
                return 0.5
        return 0.0

    def calculate_cot_reward(self, reasoning: str) -> float:
        """计算推理过程（chain-of-thought）的奖励
        
        检查推理过程是否完整且合理
        """
        # 关键词列表
        keywords = [
            '差值', '速度', '间隔', '匀速', '计算', '预测',
            '相邻', '规律', '运动'
        ]
        
        # 检查是否包含数值计算
        has_numbers = bool(re.search(r'\d+\.?\d*', reasoning))
        
        # 检查关键词出现情况
        keyword_count = sum(1 for word in keywords if word in reasoning)
        
        # 检查推理步骤是否完整
        has_steps = '1' in reasoning and '2' in reasoning and '3' in reasoning
        
        # 计算总分
        score = 0.0
        if has_numbers:
            score += 0.4  # 包含具体计算
        score += 0.3 * (keyword_count / len(keywords))  # 关键词覆盖
        if has_steps:
            score += 0.3  # 步骤完整性
            
        return score

    def calculate_trend_reward(self, predicted: str, actual: str) -> float:
        """计算预测值的变化趋势奖励
        
        检查预测值是否保持了正确的变化趋势
        """
        pred_coords = self.parse_coordinates(predicted)
        actual_coords = self.parse_coordinates(actual)
        
        if not pred_coords or not actual_coords or len(pred_coords) != len(actual_coords):
            return 0.0
            
        # 计算实际值和预测值的差分
        actual_diffs = np.diff(actual_coords)
        pred_diffs = np.diff(pred_coords)
        
        # 检查差分的符号是否一致
        correct_trends = np.sum(np.sign(actual_diffs) == np.sign(pred_diffs))
        trend_accuracy = correct_trends / len(actual_diffs)
        
        # 检查差分值的相对误差
        diff_errors = np.abs(actual_diffs - pred_diffs) / (np.abs(actual_diffs) + 1e-6)
        diff_accuracy = np.mean(np.exp(-diff_errors))
        
        return 0.5 * trend_accuracy + 0.5 * diff_accuracy

    def __call__(self, predictions: List[str], targets: List[str], reasoning: List[str]) -> torch.Tensor:
        """计算总体奖励
        
        Args:
            predictions: 模型预测的坐标序列
            targets: 实际的坐标序列
            reasoning: 模型的推理过程
        Returns:
            每个样本的总体奖励值
        """
        batch_rewards = []
        
        for pred, target, reason in zip(predictions, targets, reasoning):
            # 计算各个维度的奖励
            accuracy_reward = self.calculate_accuracy_reward(pred, target)
            format_reward = self.calculate_format_reward(pred)
            cot_reward = self.calculate_cot_reward(reason)
            trend_reward = self.calculate_trend_reward(pred, target)
            
            # 计算加权总奖励
            total_reward = (
                self.weights['accuracy'] * accuracy_reward +
                self.weights['format'] * format_reward +
                self.weights['cot'] * cot_reward +
                self.weights['trend'] * trend_reward
            )
            
            batch_rewards.append(total_reward)
            
        return torch.tensor(batch_rewards)