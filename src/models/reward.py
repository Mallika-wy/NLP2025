import numpy as np
import re
from typing import List, Union, Optional
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
    """Calculate accuracy reward for coordinate predictions
    只关注预测坐标的准确性，使用MSE评估，完全不考虑格式问题
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        kwargs: Must contain 'target' (target coordinates)
    Returns:
        List of reward values for each sample (0.0-1.0)
    """
    targets = kwargs.get('target', None)
    if not targets:
        return [0.0] * len(completions)
    
    rewards = []
    for completion, target in zip(completions, targets):
        try:
            # 首先检查基本格式是否正确
            if not all(tag in completion for tag in ['[Analysis]', '[ANSWER]', '[End]']):
                rewards.append(0.0)
                continue
                
            # 检查标签顺序是否正确
            analysis_pos = completion.find('[Analysis]')
            answer_pos = completion.find('[ANSWER]')
            end_pos = completion.find('[End]')
            
            if not (0 <= analysis_pos < answer_pos < end_pos):
                rewards.append(0.0)
                continue
         
            answer_text = completion[answer_pos:end_pos].strip()
            answer_match = re.search(r'\[ANSWER\](.*)', answer_text, re.DOTALL)

            if not answer_match:
                rewards.append(0.0)
                continue
            
            # 提取预测的坐标
            coords_text = answer_match.group(1).strip()
            coords_pattern = r'\(([-+]?\d+\.\d{1})\)'
            pred_coords = [float(x) for x in re.findall(coords_pattern, coords_text)]
            
            # 如果没有找到任何坐标，给0分
            if not pred_coords:
                rewards.append(0.0)
                continue
            
            # 提取目标坐标
            target_coords = [float(x) for x in re.findall(coords_pattern, target)]
            
            # 根据预测坐标数量计算得分
            n_pred = len(pred_coords)
            
            # 计算要比较的坐标数量
            n_compare = min(n_pred, len(target_coords), 5)
            pred_coords = pred_coords[:n_compare]
            target_coords = target_coords[:n_compare]
            
            # 计算MSE
            mse = mean_squared_error(target_coords, pred_coords)
            accuracy = np.exp(-mse)  # 转换为0-1之间的分数
            
            score = 0

            # 根据坐标数量调整分数
            if n_pred == 5:  # 正好5个坐标，满分
                score = accuracy
            elif n_pred < 5:  # 少于5个坐标，降低分数
                score = accuracy * 0.8 * (n_pred / 5)
            else:  # 多于5个坐标，也降低分数
                score = accuracy * 0.9
            
            rewards.append(float(score))

        except Exception as e:
            print(f"Error in accuracy_reward: {e}")
            rewards.append(0.0)
    
    return rewards

def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Evaluate format compliance
    只关注回答的格式规范性，不考虑内容准确性
    
    Scoring criteria:
    1. 基本结构 (0.4): 必需标签的存在和顺序
    2. 分析步骤格式 (0.3): S1-S3步骤的存在和格式
    3. 答案格式 (0.3): 坐标的格式规范
    """
    rewards = []
    
    for completion in completions:
        try:
            score = 0.0
            
            # 1. 基本结构检查 (0.4)
            # 检查标签是否存在且顺序正确
            if not all(tag in completion for tag in ['[Analysis]', '[ANSWER]', '[End]']):
                rewards.append(0.0)
                continue
                
            analysis_pos = completion.find('[Analysis]')
            answer_pos = completion.find('[ANSWER]')
            end_pos = completion.find('[End]')
            
            if not (0 <= analysis_pos < answer_pos < end_pos):
                rewards.append(0.0)
                continue
                
            score += 0.4
            
            # 2. 分析步骤格式 (0.3)
            analysis_text = completion[analysis_pos:answer_pos]
            step_score = 0.0
            
            # 检查步骤标签存在性
            if all(f"S{i}:" in analysis_text for i in range(1, 4)):
                step_score += 0.1
            
            # 检查Δx计算格式
            if re.search(r'Δx\d+\s*=.*=', analysis_text):
                step_score += 0.1
            
            # 检查速度计算格式
            if re.search(r'(Speed|Average).*=.*', analysis_text):
                step_score += 0.1
                
            score += step_score
            
            # 3. 答案格式 (0.3)
            # 提取[ANSWER]和[End]之间的内容
            answer_text = completion[answer_pos:end_pos].strip()
            answer_match = re.search(r'\[ANSWER\](.*)', answer_text, re.DOTALL)
            
            if answer_match:
                coords_text = answer_match.group(1).strip()
                # 检查坐标格式和数量
                coords = re.findall(r'\(([-+]?\d+\.\d{1})\)', coords_text)
                if len(coords) == 5:  # 正好5个坐标
                    score += 0.2
                # 检查坐标之间的分隔符格式
                if re.match(r'\s*\([-+]?\d+\.\d{1}\)(\s*,\s*\([-+]?\d+\.\d{1}\))*\s*$', coords_text):
                    score += 0.1
            
            rewards.append(float(score))
            
        except Exception as e:
            print(f"Error in format_reward: {e}")
            rewards.append(0.0)
    
    return rewards
