import numpy as np
import re
from typing import List, Union
import torch
from sklearn.metrics import mean_squared_error

def parse_coordinates(coord_str: str) -> List[float]:
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

def accuracy_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """计算预测坐标的准确性奖励
    
    使用MSE的负值作为奖励，并进行归一化
    Args:
        prompts: 输入提示列表
        completions: 模型生成的完成列表
        **kwargs: 其他参数，包括targets
    Returns:
        每个样本的奖励值列表
    """
    targets = kwargs.get('targets', [])
    if not targets:
        return [0.0] * len(completions)
        
    rewards = []
    for completion, target in zip(completions, targets):
        pred_coords = parse_coordinates(completion)
        actual_coords = parse_coordinates(target)
        
        if not pred_coords or not actual_coords or len(pred_coords) != len(actual_coords):
            rewards.append(-1.0)
            continue
            
        # 计算MSE
        mse = mean_squared_error(actual_coords, pred_coords)
        # 将MSE转换为0到1之间的奖励值
        reward = np.exp(-mse)  # 使用指数函数将MSE转换为奖励
        rewards.append(reward)
    
    return rewards

def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """计算输出格式的正确性奖励
    
    检查是否符合 "(x.xx)" 的格式要求
    Args:
        prompts: 输入提示列表
        completions: 模型生成的完成列表
        **kwargs: 其他参数
    Returns:
        每个样本的奖励值列表
    """
    rewards = []
    pattern = r'^\s*(\([-+]?\d*\.?\d+\)[\s,]*)+\s*$'
    
    for completion in completions:
        if re.match(pattern, completion):
            # 检查是否包含正确数量的坐标（8个）
            coords = re.findall(r'\(([-+]?\d*\.?\d+)\)', completion)
            rewards.append(1.0 if len(coords) == 8 else 0.5)
        else:
            rewards.append(0.0)
    
    return rewards

def cot_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """计算推理过程（chain-of-thought）的奖励
    
    检查推理过程是否完整且合理
    Args:
        prompts: 输入提示列表
        completions: 模型生成的完成列表
        **kwargs: 其他参数，包括reasoning
    Returns:
        每个样本的奖励值列表
    """
    reasoning_list = kwargs.get('reasoning', [])
    if not reasoning_list:
        return [0.0] * len(completions)
        
    keywords = ['差值', '速度', '间隔', '匀速', '计算', '预测', '相邻', '规律', '运动']
    rewards = []
    
    for reasoning in reasoning_list:
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
            
        rewards.append(score)
    
    return rewards

def trend_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """计算预测值的变化趋势奖励
    
    检查预测值是否保持了正确的变化趋势
    Args:
        prompts: 输入提示列表
        completions: 模型生成的完成列表
        **kwargs: 其他参数，包括targets
    Returns:
        每个样本的奖励值列表
    """
    targets = kwargs.get('targets', [])
    if not targets:
        return [0.0] * len(completions)
        
    rewards = []
    for completion, target in zip(completions, targets):
        pred_coords = parse_coordinates(completion)
        actual_coords = parse_coordinates(target)
        
        if not pred_coords or not actual_coords or len(pred_coords) != len(actual_coords):
            rewards.append(0.0)
            continue
            
        # 计算实际值和预测值的差分
        actual_diffs = np.diff(actual_coords)
        pred_diffs = np.diff(pred_coords)
        
        # 检查差分的符号是否一致
        correct_trends = np.sum(np.sign(actual_diffs) == np.sign(pred_diffs))
        trend_accuracy = correct_trends / len(actual_diffs)
        
        # 检查差分值的相对误差
        diff_errors = np.abs(actual_diffs - pred_diffs) / (np.abs(actual_diffs) + 1e-6)
        diff_accuracy = np.mean(np.exp(-diff_errors))
        
        rewards.append(0.5 * trend_accuracy + 0.5 * diff_accuracy)
    
    return rewards