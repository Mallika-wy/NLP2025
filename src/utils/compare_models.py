import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from evaluate import evaluate_predictions

def plot_comparison_bar(metrics_05B: Dict, metrics_15B: Dict, metric_name: str, 
                       title: str, save_path: str):
    """绘制对比柱状图"""
    labels = ['0.5B Model', '1.5B Model']
    values = [metrics_05B[metric_name], metrics_15B[metric_name]]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['lightblue', 'lightgreen'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_position_errors(metrics_05B: Dict, metrics_15B: Dict, save_path: str):
    """绘制位置误差对比图"""
    # 获取两个模型共有的位置
    positions_05B = set(metrics_05B['position_metrics'].keys())
    positions_15B = set(metrics_15B['position_metrics'].keys())
    common_positions = sorted(positions_05B.intersection(positions_15B))
    
    if not common_positions:
        print("警告：没有找到两个模型共同的预测位置")
        return
        
    errors_05B = [metrics_05B['position_metrics'][pos]['mean_error'] for pos in common_positions]
    errors_15B = [metrics_15B['position_metrics'][pos]['mean_error'] for pos in common_positions]
    
    plt.figure(figsize=(10, 6))
    plt.plot(common_positions, errors_05B, 'o-', label='0.5B Model', color='lightblue')
    plt.plot(common_positions, errors_15B, 'o-', label='1.5B Model', color='lightgreen')
    plt.xlabel('Position')
    plt.ylabel('Mean Error')
    plt.title('Position-wise Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    
    # 打印位置信息
    print(f"\n位置误差对比:")
    print("-" * 50)
    print(f"{'Position':8} {'0.5B':>10} {'1.5B':>10} {'改进比例':>10}")
    print("-" * 50)
    for pos, err_05B, err_15B in zip(common_positions, errors_05B, errors_15B):
        improvement = (err_05B - err_15B) / err_05B * 100
        print(f"{pos:8d} {err_05B:>10.4f} {err_15B:>10.4f} {improvement:>10.2f}%")

def compare_models():
    # 模型路径
    model_paths = {
        '0.5B': {
            'in_distribution': 'results/0.5B-5-200/test_in.json',
            'out_of_distribution': 'results/0.5B-5-200/test_out.json'
        },
        '1.5B': {
            'in_distribution': 'results/1.5B-5-200/test_in.json',
            'out_of_distribution': 'results/1.5B-5-200/test_out.json'
        }
    }
    
    # 创建输出目录
    output_dir = 'results/model_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储评估结果
    results = {}
    
    # 评估每个模型在不同数据集上的表现
    for model_name, paths in model_paths.items():
        results[model_name] = {}
        for dataset_type, file_path in paths.items():
            if not os.path.exists(file_path):
                print(f"警告: {file_path} 不存在")
                continue
            
            metrics = evaluate_predictions(file_path, os.path.join(output_dir, f'{model_name}_{dataset_type}'))
            results[model_name][dataset_type] = metrics
    
    # 生成对比图表
    for dataset_type in ['in_distribution', 'out_of_distribution']:
        metrics_05B = results['0.5B'][dataset_type]
        metrics_15B = results['1.5B'][dataset_type]
        
        # 绘制MSE对比图
        plot_comparison_bar(
            metrics_05B, metrics_15B, 'average_mse',
            f'MSE Comparison ({dataset_type})',
            os.path.join(output_dir, f'mse_comparison_{dataset_type}.png')
        )
        
        # 绘制MAE对比图
        plot_comparison_bar(
            metrics_05B, metrics_15B, 'average_mae',
            f'MAE Comparison ({dataset_type})',
            os.path.join(output_dir, f'mae_comparison_{dataset_type}.png')
        )
        
        # 绘制准确率对比图
        plot_comparison_bar(
            metrics_05B, metrics_15B, 'sequence_accuracy',
            f'Sequence Accuracy Comparison ({dataset_type})',
            os.path.join(output_dir, f'accuracy_comparison_{dataset_type}.png')
        )
        
        # 绘制位置误差对比图
        plot_position_errors(
            metrics_05B, metrics_15B,
            os.path.join(output_dir, f'position_errors_{dataset_type}.png')
        )
        
        # 打印详细对比结果
        print(f"\n{dataset_type} 数据集对比结果:")
        print("=" * 50)
        print(f"{'指标':20} {'0.5B':>10} {'1.5B':>10} {'提升比例':>10}")
        print("-" * 50)
        
        metrics_to_compare = {
            'MSE': 'average_mse',
            'MAE': 'average_mae',
            'Sequence Accuracy': 'sequence_accuracy',
            'Length Accuracy': 'length_accuracy'
        }
        
        for metric_name, metric_key in metrics_to_compare.items():
            value_05B = metrics_05B[metric_key]
            value_15B = metrics_15B[metric_key]
            
            if metric_name in ['MSE', 'MAE']:  # 对于误差指标，减少是改进
                improvement = (value_05B - value_15B) / value_05B * 100
                print(f"{metric_name:20} {value_05B:>10.4f} {value_15B:>10.4f} {improvement:>10.2f}%")
            else:  # 对于准确率指标，增加是改进
                improvement = (value_15B - value_05B) / value_05B * 100
                print(f"{metric_name:20} {value_05B:>10.2%} {value_15B:>10.2%} {improvement:>10.2f}%")
        
        # 打印误差分布对比
        print("\n误差分布对比:")
        print("-" * 50)
        percentiles = ['percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'percentile_95']
        for p in percentiles:
            value_05B = metrics_05B['error_distribution'][p]
            value_15B = metrics_15B['error_distribution'][p]
            improvement = (value_05B - value_15B) / value_05B * 100
            print(f"{p:20} {value_05B:>10.4f} {value_15B:>10.4f} {improvement:>10.2f}%")
    
    # 保存完整的对比结果
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    compare_models() 