import json
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.dataset import MotionDataset
from src.models.reward import RewardFunction
from src.utils.visualization import TrainingVisualizer
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TestEvaluator:
    def __init__(self, model, tokenizer, reward_fn, visualizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.visualizer = visualizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def evaluate_batch(self, batch):
        """评估单个批次"""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        # 解码预测结果
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        targets = batch["response"]
        
        # 计算奖励
        rewards = self.reward_fn(predictions, targets, [""] * len(predictions))
        
        return predictions, rewards
    
    def evaluate_dataset(self, dataset, split_name):
        """评估整个数据集"""
        all_rewards = []
        all_predictions = []
        all_targets = []
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            predictions, rewards = self.evaluate_batch(batch)
            all_rewards.extend(rewards.cpu().numpy())
            all_predictions.extend(predictions)
            all_targets.extend(batch["response"])
            
        return {
            "rewards": all_rewards,
            "predictions": all_predictions,
            "targets": all_targets,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards)
        }
    
    def visualize_results(self, in_dist_results, out_dist_results):
        """可视化测试结果"""
        # 奖励分布对比图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=in_dist_results["rewards"], label="In-distribution")
        sns.kdeplot(data=out_dist_results["rewards"], label="Out-of-distribution")
        plt.xlabel("Reward")
        plt.ylabel("Density")
        plt.title("Reward Distribution Comparison")
        plt.legend()
        wandb.log({"test/reward_distribution": wandb.Image(plt)})
        plt.close()
        
        # 生成详细的测试报告
        test_report = {
            "in_distribution": {
                "mean_reward": in_dist_results["mean_reward"],
                "std_reward": in_dist_results["std_reward"],
                "sample_predictions": list(zip(
                    in_dist_results["predictions"][:5],
                    in_dist_results["targets"][:5]
                ))
            },
            "out_of_distribution": {
                "mean_reward": out_dist_results["mean_reward"],
                "std_reward": out_dist_results["std_reward"],
                "sample_predictions": list(zip(
                    out_dist_results["predictions"][:5],
                    out_dist_results["targets"][:5]
                ))
            }
        }
        
        return test_report

def main():
    # 加载配置
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化wandb
    wandb.init(
        project="motion-prediction-rl",
        config=config,
        name=config.get("experiment_name", "motion-prediction-testing")
    )
    
    # 初始化可视化工具
    visualizer = TrainingVisualizer(config)
    
    # 加载模型和tokenizer
    model_name = config["model"]["name"]
    model_path = config["testing"]["model_checkpoint_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 初始化奖励函数
    reward_fn = RewardFunction(config, visualizer)
    
    # 创建评估器
    evaluator = TestEvaluator(model, tokenizer, reward_fn, visualizer)
    
    # 加载测试数据
    with open("data/test_in/motion_data.json", "r") as f:
        test_in_data = json.load(f)
    with open("data/test_out/motion_data.json", "r") as f:
        test_out_data = json.load(f)
    
    # 创建测试数据集
    test_in_dataset = MotionDataset(test_in_data, model_name)
    test_out_dataset = MotionDataset(test_out_data, model_name)
    
    # 评估模型
    print("Evaluating in-distribution data...")
    in_dist_results = evaluator.evaluate_dataset(test_in_dataset, "in-distribution")
    
    print("Evaluating out-of-distribution data...")
    out_dist_results = evaluator.evaluate_dataset(test_out_dataset, "out-of-distribution")
    
    # 可视化结果
    test_report = evaluator.visualize_results(in_dist_results, out_dist_results)
    
    # 保存测试报告
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "test_report.json", "w") as f:
        json.dump(test_report, f, indent=2)
    
    # 打印主要结果
    print("\nTest Results Summary:")
    print("In-distribution:")
    print(f"Mean Reward: {in_dist_results['mean_reward']:.4f}")
    print(f"Std Reward: {in_dist_results['std_reward']:.4f}")
    print("\nOut-of-distribution:")
    print(f"Mean Reward: {out_dist_results['mean_reward']:.4f}")
    print(f"Std Reward: {out_dist_results['std_reward']:.4f}")
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    main()