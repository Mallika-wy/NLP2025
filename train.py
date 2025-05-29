import json
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer
from src.data.dataset import MotionDataset
from src.models.reward import RewardFunction
from src.utils.visualization import TrainingVisualizer
import yaml

def main():
    # 加载配置
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化wandb
    wandb.init(
        project="motion-prediction-rl",
        config=config,
        name=config.get("experiment_name", "motion-prediction-training")
    )
    
    # 初始化可视化工具
    visualizer = TrainingVisualizer(config)
    
    # 加载数据
    with open("data/train/motion_data.json", "r") as f:
        train_data = json.load(f)
    
    # 初始化tokenizer和模型
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 创建数据集
    train_dataset = MotionDataset(train_data, model_name)
    
    # 初始化奖励函数
    reward_fn = RewardFunction(config, visualizer)
    
    # 设置训练器
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_function=reward_fn,
        args=config["training"]
    )
    
    # 开始训练
    trainer.train()
    
    # 生成训练分析报告
    visualizer.plot_rewards_history()
    visualizer.plot_reward_distribution()
    visualizer.plot_prediction_accuracy()
    training_report = visualizer.generate_training_report()
    
    # 保存训练报告
    with open("training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    main()