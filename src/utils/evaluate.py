import json
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Tuple
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_coordinates(text: str) -> List[float]:
    """从文本中提取坐标值
    
    Args:
        text: 包含坐标的文本
    Returns:
        坐标列表
    """
    coords_pattern = r'\(([-+]?\d+\.\d{1})\)'  # 匹配一位小数的坐标
    coords = re.findall(coords_pattern, text)
    return [float(x) for x in coords]

def calculate_error_distribution(errors: List[float]) -> Dict[str, float]:
    """计算误差分布
    
    Args:
        errors: 误差列表
    Returns:
        误差分布统计
    """
    percentiles = [25, 50, 75, 90, 95]
    distribution = {
        f'percentile_{p}': float(np.percentile(errors, p))
        for p in percentiles
    }
    distribution['mean'] = float(np.mean(errors))
    distribution['std'] = float(np.std(errors))
    return distribution

def plot_error_distribution(errors: List[float], title: str, save_path: str):
    """绘制误差分布图
    
    Args:
        errors: 误差列表
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(errors), color='r', linestyle='dashed', linewidth=2)
    plt.title(title)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def evaluate_predictions(predictions_file: str, output_dir: str) -> Dict:
    """评估预测结果
    
    Args:
        predictions_file: 预测结果文件路径
        output_dir: 输出目录
    Returns:
        评估指标字典
    """
    # 加载预测结果
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    targets = data['targets']
    
    all_mse = []
    all_mae = []
    all_errors = []  # 存储所有误差
    position_errors = defaultdict(list)  # 按位置存储误差
    sequence_accuracy = 0
    total_sequences = len(predictions)
    
    for pred, target in zip(predictions, targets):
        # 提取预测坐标和目标坐标
        pred_coords = extract_coordinates(pred['predicted_coordinates'])
        target_coords = extract_coordinates(target)
        
        # 确保比较相同数量的坐标
        n_coords = min(len(pred_coords), len(target_coords))
        if n_coords == 0:
            continue
            
        pred_coords = pred_coords[:n_coords]
        target_coords = target_coords[:n_coords]
        
        # 计算每个位置的误差
        errors = np.abs(np.array(pred_coords) - np.array(target_coords))
        for pos, error in enumerate(errors):
            position_errors[pos].append(error)
            all_errors.append(error)
        
        # 计算MSE和MAE
        mse = mean_squared_error(target_coords, pred_coords)
        mae = mean_absolute_error(target_coords, pred_coords)
        
        all_mse.append(mse)
        all_mae.append(mae)
        
        # 检查是否完全匹配（允许小误差）
        if np.all(errors < 0.1):
            sequence_accuracy += 1
    
    # 计算平均指标
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    accuracy = sequence_accuracy / total_sequences
    
    # 计算预测长度的准确率
    length_accuracy = sum(1 for pred, target in zip(predictions, targets)
                         if len(extract_coordinates(pred['predicted_coordinates'])) == 
                         len(extract_coordinates(target))) / total_sequences
    
    # 计算误差分布
    error_distribution = calculate_error_distribution(all_errors)
    
    # 计算每个位置的平均误差
    position_metrics = {
        pos: {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors))
        }
        for pos, errors in position_errors.items()
    }
    
    # 生成可视化
    os.makedirs(output_dir, exist_ok=True)
    plot_error_distribution(
        all_errors,
        'Error Distribution',
        os.path.join(output_dir, 'error_distribution.png')
    )
    
    return {
        'average_mse': avg_mse,
        'average_mae': avg_mae,
        'sequence_accuracy': accuracy,
        'length_accuracy': length_accuracy,
        'total_sequences': total_sequences,
        'error_distribution': error_distribution,
        'position_metrics': position_metrics
    }

def main():
    # 评估in-distribution和out-of-distribution结果
    results_paths = {
        'in_distribution': 'results/1.5B-5-200/test_in.json',
        'out_of_distribution': 'results/1.5B-5-200/test_out.json'
    }
    
    for dataset_type, file_path in results_paths.items():
        if not os.path.exists(file_path):
            print(f"警告: {file_path} 不存在")
            continue
            
        print(f"\n评估 {dataset_type} 数据集:")
        output_dir = os.path.join('results', f'evaluation_{dataset_type}')
        metrics = evaluate_predictions(file_path, output_dir)
        
        print(f"总序列数: {metrics['total_sequences']}")
        print(f"平均MSE: {metrics['average_mse']:.4f}")
        print(f"平均MAE: {metrics['average_mae']:.4f}")
        print(f"序列完全匹配率: {metrics['sequence_accuracy']:.2%}")
        print(f"预测长度准确率: {metrics['length_accuracy']:.2%}")
        
        print("\n误差分布:")
        for k, v in metrics['error_distribution'].items():
            print(f"{k}: {v:.4f}")
        
        print("\n各位置误差:")
        for pos, pos_metrics in metrics['position_metrics'].items():
            print(f"位置 {pos}:")
            print(f"  平均误差: {pos_metrics['mean_error']:.4f}")
            print(f"  误差标准差: {pos_metrics['std_error']:.4f}")
        
        # 保存详细结果
        with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
