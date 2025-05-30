import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import pandas as pd
from pathlib import Path
import json
import os

class TrainingVisualizer:
    def __init__(self, config):
        """初始化可视化工具
        
        Args:
            config: 配置字典，包含可视化相关的参数
        """
        self.config = config
        self.rewards_history = {
            'total_rewards': [],
            'accuracy_rewards': [],
            'format_rewards': [],
            'cot_rewards': [],
            'trend_rewards': []
        }
        self.prediction_samples = []
        self.step_count = 0
        
        # 创建保存目录
        self.viz_dir = Path("visualization_results")
        self.viz_dir.mkdir(exist_ok=True)
        
        # 设置可视化样式
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        # 设置图表参数
        self.figure_size = config["visualization"]["plotting"].get("figure_size", [12, 6])
        self.dpi = config["visualization"]["plotting"].get("dpi", 100)
    
    def log_rewards(self, rewards_dict: Dict[str, float], step: int):
        """记录每个步骤的奖励值
        
        Args:
            rewards_dict: 包含各类奖励值的字典
            step: 当前步数
        """
        for key, value in rewards_dict.items():
            self.rewards_history[key].append(value)
            wandb.log({f"rewards/{key}": value}, step=step)
        self.step_count = step
    
    def log_prediction_sample(self, 
                            input_coords: str, 
                            predicted_coords: str, 
                            actual_coords: str,
                            total_reward: float,
                            step: int):
        """记录预测样本
        
        Args:
            input_coords: 输入坐标序列
            predicted_coords: 预测的坐标序列
            actual_coords: 实际的坐标序列
            total_reward: 总奖励值
            step: 当前步数
        """
        self.prediction_samples.append({
            'step': step,
            'input': input_coords,
            'predicted': predicted_coords,
            'actual': actual_coords,
            'reward': total_reward
        })
        
        # 记录到wandb
        wandb.log({
            "predictions/sample": {
                "input": input_coords,
                "predicted": predicted_coords,
                "actual": actual_coords,
                "reward": total_reward
            }
        }, step=step)
    
    def plot_rewards_history(self):
        """绘制奖励历史趋势图"""
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        for key, values in self.rewards_history.items():
            if values:  # 只绘制非空的奖励历史
                plt.plot(range(len(values)), values, label=key, alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Reward Value')
        plt.title('Training Rewards History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig(self.viz_dir / "rewards_history.png")
        wandb.log({"charts/rewards_history": wandb.Image(plt)})
        plt.close()
    
    def plot_reward_distribution(self):
        """绘制奖励分布图"""
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        data = pd.DataFrame(self.rewards_history)
        sns.boxplot(data=data)
        plt.xticks(rotation=45)
        plt.title('Reward Distribution by Type')
        
        # 保存图片
        plt.savefig(self.viz_dir / "reward_distribution.png")
        wandb.log({"charts/reward_distribution": wandb.Image(plt)})
        plt.close()
    
    def plot_prediction_accuracy(self):
        """绘制预测准确性散点图"""
        if not self.prediction_samples:
            return
            
        df = pd.DataFrame(self.prediction_samples)
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        sns.scatterplot(data=df, x='step', y='reward', alpha=0.5)
        plt.xlabel('Training Steps')
        plt.ylabel('Prediction Reward')
        plt.title('Prediction Accuracy Over Time')
        
        # 添加趋势线
        z = np.polyfit(df['step'], df['reward'], 1)
        p = np.poly1d(z)
        plt.plot(df['step'], p(df['step']), "r--", alpha=0.8, label='Trend')
        plt.legend()
        
        # 保存图片
        plt.savefig(self.viz_dir / "prediction_accuracy.png")
        wandb.log({"charts/prediction_accuracy": wandb.Image(plt)})
        plt.close()
    
    def generate_training_report(self):
        """生成训练报告"""
        report = {
            "training_steps": self.step_count,
            "average_rewards": {
                key: np.mean(values) if values else 0
                for key, values in self.rewards_history.items()
            },
            "reward_std": {
                key: np.std(values) if values else 0
                for key, values in self.rewards_history.items()
            },
            "best_predictions": sorted(
                self.prediction_samples,
                key=lambda x: x['reward'],
                reverse=True
            )[:5] if self.prediction_samples else []
        }
        
        # 保存报告
        with open(self.viz_dir / "training_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 记录到wandb
        wandb.log({"training_report": report})
        
        return report
    
    def plot_learning_curves(self):
        """绘制学习曲线"""
        if not self.rewards_history['total_rewards']:
            return
            
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # 计算移动平均
        window_size = 50
        rewards = np.array(self.rewards_history['total_rewards'])
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # 绘制原始奖励和移动平均
        plt.plot(rewards, alpha=0.3, label='Raw Rewards')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Total Reward')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig(self.viz_dir / "learning_curves.png")
        wandb.log({"charts/learning_curves": wandb.Image(plt)})
        plt.close()
    
    def save_visualization_state(self):
        """保存可视化状态"""
        state = {
            'rewards_history': self.rewards_history,
            'prediction_samples': self.prediction_samples,
            'step_count': self.step_count
        }
        
        with open(self.viz_dir / "visualization_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_visualization_state(self):
        """加载可视化状态"""
        try:
            with open(self.viz_dir / "visualization_state.json", "r", encoding="utf-8") as f:
                state = json.load(f)
                self.rewards_history = state['rewards_history']
                self.prediction_samples = state['prediction_samples']
                self.step_count = state['step_count']
        except FileNotFoundError:
            print("No previous visualization state found.")