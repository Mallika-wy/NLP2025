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

def accuracy_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """计算坐标预测的准确性奖励
    
    Args:
        prompts: 输入提示列表
        completions: 模型生成的完成列表
        kwargs: 必须包含target（目标坐标）
    Returns:
        每个样本的奖励值列表
    """
    targets = kwargs.get('target', None)
    if not targets:
        return [0.0] * len(completions)
    
    rewards = []
    for completion, target in zip(completions, targets):
        try:
            # 1. 首先检查是否有Coordinates部分和End部分
            coords_section = re.search(r'\[Coordinates\](.*?)\[End\]', completion, re.DOTALL)
            if not coords_section:
                rewards.append(0.0)
                continue
            
            # 2. 提取预测的坐标
            coords_text = coords_section.group(1)
            coords_pattern = r'\(([-+]?\d+\.\d{1})\)'
            pred_coords = [float(x) for x in re.findall(coords_pattern, coords_text)]
            
            # 如果没有找到任何坐标，给0分
            if not pred_coords:
                rewards.append(0.0)
                continue
            
            # 3. 提取目标坐标
            target_coords = [float(x) for x in re.findall(coords_pattern, target)]
            
            # 4. 根据预测坐标数量计算得分
            n_pred = len(pred_coords)
            base_score = 0.0
            
            # 计算要比较的坐标数量
            n_compare = min(n_pred, len(target_coords), 5)
            pred_coords = pred_coords[:n_compare]
            target_coords = target_coords[:n_compare]
            
            # 计算MSE
            mse = mean_squared_error(target_coords, pred_coords)
            accuracy = np.exp(-mse)  # 转换为0-1之间的分数
            
            # 根据坐标数量调整分数
            if n_pred == 5:  # 正好5个坐标，满分
                base_score = accuracy
            elif n_pred < 5:  # 少于5个坐标，降低分数
                base_score = accuracy * 0.8 * (n_pred / 5)
            else:  # 多于5个坐标，也降低分数
                base_score = accuracy * 0.9
            
            # 额外的精度奖励（一位小数）
            if all(len(str(c).split('.')[-1]) == 1 for c in pred_coords):
                base_score *= 1.1
            
            rewards.append(float(base_score))
            
        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            rewards.append(0.0)
    
    return rewards

def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """评估回答的格式规范性
    
    评分标准:
    1. 基本结构 (0.3): [Analysis], [Coordinates] 和 [End] 标题及其顺序
    2. 分析简洁性 (0.3): 
       - S1回答是否简洁（仅包含Yes/No和一行解释）
       - S2是否使用正确格式 (Speed = x.x)
    3. 坐标格式 (0.4): 正确的坐标格式和数量
    """
    rewards = []
    
    # 最大允许的字符数
    MAX_S1_LENGTH = 50  # S1步骤的最大字符数
    MAX_S2_LENGTH = 20  # S2步骤的最大字符数
    
    for completion in completions:
        score = 0.0
        
        # 1. 基本结构检查 (0.3分)
        analysis_pos = completion.find('[Analysis]')
        coords_pos = completion.find('[Coordinates]')
        end_pos = completion.find('[End]')
        
        if all(pos != -1 for pos in [analysis_pos, coords_pos, end_pos]):
            if analysis_pos < coords_pos < end_pos:  # 顺序正确
                score += 0.3
            else:  # 顺序错误但结构完整
                score += 0.05
        
        # 2. 分析简洁性检查 (0.3分)
        if analysis_pos != -1:
            analysis_text = completion[analysis_pos:coords_pos if coords_pos != -1 else None]
            
            # S1检查 (0.15分)
            s1_match = re.search(r'S1:(.*?)(?=S2:|$)', analysis_text, re.DOTALL)
            if s1_match:
                s1_text = s1_match.group(1).strip()
                if len(s1_text) <= MAX_S1_LENGTH and ('Yes' in s1_text or 'No' in s1_text):
                    score += 0.15
            
            # S2检查 (0.15分)
            s2_match = re.search(r'S2:(.*?)(?=\[Coordinates\]|$)', analysis_text, re.DOTALL)
            if s2_match:
                s2_text = s2_match.group(1).strip()
                if len(s2_text) <= MAX_S2_LENGTH and re.search(r'Speed\s*=\s*[-+]?\d+\.\d{1}', s2_text):
                    score += 0.15
        
        # 3. 坐标格式检查 (0.4分)
        if coords_pos != -1 and end_pos != -1:
            coords_text = completion[coords_pos:end_pos]
            coords = re.findall(r'\(([-+]?\d+\.\d{1})\)', coords_text)
            
            # 检查坐标数量
            if len(coords) == 5:
                score += 0.2
                
            # 检查坐标格式
            if coords and all(re.match(r'^[-+]?\d+\.\d{1}$', c) for c in coords):
                score += 0.2
        
        rewards.append(float(score))
    
    return rewards
