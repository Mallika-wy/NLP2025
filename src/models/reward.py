import numpy as np
import re
from typing import List, Union, Optional
import torch
from sklearn.metrics import mean_squared_error

def parse_coordinates(text: str) -> tuple[List[float], str]:
    """解析文本中的坐标和推理过程
    Args:
        text: 包含坐标和推理过程的文本
    Returns:
        coords: 坐标列表
        reasoning: 推理过程
    """
    # 提取坐标
    coords_pattern = r'\(([-+]?\d*\.?\d+)\)'
    coords = re.findall(coords_pattern, text)
    coords = [float(x) for x in coords]
    
    # 提取推理过程
    splits = text.split('\n\n')
    reasoning = splits[1] if len(splits) > 1 else ""
    
    return coords, reasoning

def extract_step_info(reasoning: str) -> dict:
    """Extract key information from reasoning process"""
    info = {
        'has_step_markers': bool(re.search(r'S[123]:|Step|Pattern|Calculation|Prediction', reasoning.lower())),
        'has_calculations': bool(re.search(r'[-+]?\d*\.?\d+\s*[-+*/=]\s*[-+]?\d*\.?\d+', reasoning)),
        'has_analysis': bool(re.search(r'pattern|analysis|calculate|predict', reasoning.lower())),
        'mentions_speed': bool(re.search(r'speed|uniform|motion|constant', reasoning.lower())),
        'has_differences': bool(re.search(r'diff|interval|gap|increase', reasoning.lower()))
    }
    return info

def accuracy_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Calculate accuracy reward for predicted coordinates
    
    Args:
        prompts: List of input prompts
        completions: List of model generated completions
        kwargs: Must contain 'target' with target coordinates
    Returns:
        List of reward values for each sample
    """
    targets = kwargs.get('target', None)
    if not targets:
        return [0.0] * len(completions)
    
    rewards = []
    for completion, target in zip(completions, targets):
        try:
            # 尝试找到 [Coordinates] 部分
            coords_section = re.search(r'\[Coordinates\](.*?)\[Analysis\]', completion, re.DOTALL)
            if not coords_section:
                rewards.append(-1.0)
                continue
                
            # 从坐标部分提取数字
            coords_text = coords_section.group(1)
            coords_pattern = r'\(([-+]?\d*\.\d{2})\)'
            pred_coords = [float(x) for x in re.findall(coords_pattern, coords_text)]
            
            # 从target中提取实际坐标
            actual_coords, _ = parse_coordinates(target)
            
            # 确保有足够的预测坐标
            if not pred_coords or len(pred_coords) < 5:
                rewards.append(-1.0)
                continue
                
            # 只取前5个坐标进行比较
            pred_coords = pred_coords[:5]
            actual_coords = actual_coords[:5]
            
            # 计算MSE并转换为奖励
            mse = mean_squared_error(actual_coords, pred_coords)
            reward = np.exp(-mse)  # 使用指数函数将MSE转换为奖励
            
            # 添加额外的精度奖励：如果所有数字都精确到两位小数，给予额外奖励
            if all(len(str(c).split('.')[-1]) == 2 for c in pred_coords):
                reward *= 1.1  # 增加10%的奖励
                
            rewards.append(float(reward))  # 确保返回Python float
            
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            rewards.append(-1.0)
    
    return rewards


def cot_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Compute chain-of-thought reward
    
    Evaluate completeness and quality of reasoning process
    """
    rewards = []
    
    for completion in completions:
        _, reasoning = parse_coordinates(completion)
        if not reasoning:
            rewards.append(0.0)
            continue
            
        step_info = extract_step_info(reasoning)
        score = 0.0
        
        # 1. Steps completeness (0.3)
        if step_info['has_step_markers']:
            score += 0.3
            
        # 2. Calculation process (0.3)
        if step_info['has_calculations']:
            score += 0.3
            
        # 3. Analysis quality (0.4)
        analysis_score = 0.0
        if step_info['has_analysis']:
            analysis_score += 0.15
        if step_info['mentions_speed']:
            analysis_score += 0.15
        if step_info['has_differences']:
            analysis_score += 0.1
        score += analysis_score
        
        rewards.append(score)
    
    return rewards


def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Evaluate response format and structure
    
    Scoring criteria:
    1. Basic structure (0.2): [Coordinates] and [Analysis] headers
    2. Coordinates format (0.3): Exactly 5 coordinates in (x.xx) format
    3. Analysis steps (0.4): S1-S4 with correct labels
    4. Overall formatting (0.1): Proper spacing and newlines
    """
    rewards = []
    
    # 定义期望的格式模式
    patterns = {
        # 基本结构
        'coords_header': r'\[Coordinates\]',
        'analysis_header': r'\[Analysis\]',
        
        # 坐标格式
        'coords_format': r'\([-+]?\d+\.\d{2}\)',
        
        # 分析步骤
        'step_patterns': {
            'S1': r'S1:\s*.*observation',
            'S2': r'S2:\s*.*calculation',
            'S3': r'S3:\s*.*verification',
            'S4': r'S4:\s*.*prediction',
        }
    }
    
    for completion in completions:
        score = 0.0
        
        # 1. 检查基本结构 (0.2分)
        if re.search(patterns['coords_header'], completion, re.IGNORECASE):
            score += 0.1
        if re.search(patterns['analysis_header'], completion, re.IGNORECASE):
            score += 0.1
            
        # 2. 检查坐标格式 (0.3分)
        coords = re.findall(patterns['coords_format'], completion)
        if coords:  # 有坐标
            if len(coords) == 5:  # 正好5个坐标
                score += 0.2
            # 检查格式是否正确（包括逗号分隔）
            coords_line = re.search(r'\[Coordinates\].*?\n(.*?)\n', completion, re.DOTALL)
            if coords_line and re.match(r'\s*\([-+]?\d+\.\d{2}\)(\s*,\s*\([-+]?\d+\.\d{2}\))*\s*$', coords_line.group(1)):
                score += 0.1
                
        # 3. 检查分析步骤 (0.4分)
        for step, pattern in patterns['step_patterns'].items():
            if re.search(pattern, completion, re.IGNORECASE):
                score += 0.1
                
        # 4. 检查整体格式 (0.1分)
        # 检查是否有适当的空行和缩进
        if re.search(r'\[Coordinates\]\n.*?\n\n\[Analysis\]', completion, re.DOTALL):
            score += 0.05
        # 检查是否以换行符结束
        if completion.strip().endswith('\n'):
            score += 0.05
            
        rewards.append(float(score))  # 确保返回Python float
    
    return rewards


# def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
#     """计算输出格式的正确性奖励
    
#     检查格式规范性和一致性
#     Args:
#         prompts: 输入提示列表
#         completions: 模型生成的完成列表
#     Returns:
#         每个样本的奖励值列表
#     """
#     rewards = []
    
#     # 标准格式模式
#     coord_pattern = r'\(([-+]?\d*\.\d{2})\)'  # 要求两位小数
#     full_pattern = f'^\\s*{coord_pattern}(,\\s*{coord_pattern})*\\s*$'
    
#     for completion in completions:
#         score = 0.0
#         coords, reasoning = parse_coordinates(completion)
        
#         # 1. 基本格式检查 (0.4分)
#         if re.match(full_pattern, completion.split('\n')[0]):
#             score += 0.4
        
#         # 2. 坐标数量检查 (0.2分)
#         if len(coords) == 8:  # 要求正好8个坐标
#             score += 0.2
        
#         # 3. 格式一致性检查 (0.2分)
#         if all(len(str(c).split('.')[-1]) == 2 for c in coords):  # 检查是否都是两位小数
#             score += 0.2
            
#         # 4. 分隔符一致性 (0.2分)
#         if re.match(r'^[^,]*(?:,\s+\([^,]*\))*$', completion.split('\n')[0]):
#             score += 0.2
            
#         rewards.append(score)
    
#     return rewards


# def trend_reward(prompts: List[str], completions: List[str], targets: Optional[List[str]] = None, **kwargs) -> List[float]:
#     """计算预测值的变化趋势奖励
    
#     评估预测值的变化趋势是否符合物理规律
#     Args:
#         prompts: 输入提示列表
#         completions: 模型生成的完成列表
#         targets: 目标坐标列表
#     Returns:
#         每个样本的奖励值列表
#     """
#     if not targets:
#         return [0.0] * len(completions)
        
#     rewards = []
#     for completion, target in zip(completions, targets):
#         pred_coords, _ = parse_coordinates(completion)
#         actual_coords, _ = parse_coordinates(target)
        
#         if not pred_coords or not actual_coords or len(pred_coords) != len(actual_coords):
#             rewards.append(0.0)
#             continue
            
#         score = 0.0
        
#         # 1. 趋势一致性 (0.4分)
#         pred_diffs = np.diff(pred_coords)
#         actual_diffs = np.diff(actual_coords)
#         trend_match = np.sum(np.sign(pred_diffs) == np.sign(actual_diffs)) / len(pred_diffs)
#         score += 0.4 * trend_match
        
#         # 2. 变化率一致性 (0.3分)
#         pred_rates = np.diff(pred_diffs)
#         actual_rates = np.diff(actual_diffs)
#         if len(pred_rates) > 0:
#             rate_consistency = np.mean(np.abs(pred_rates) < 1e-5)  # 检查是否接近匀速
#             score += 0.3 * rate_consistency
        
#         # 3. 物理合理性 (0.3分)
#         if all(np.abs(diff - pred_diffs[0]) < 1e-5 for diff in pred_diffs):  # 检查是否严格匀速
#             score += 0.3
            
#         rewards.append(float(score))  # 确保返回Python float
    
#     return rewards