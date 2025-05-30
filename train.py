import os
import json
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOTrainer
from src.data.dataset import MotionDataset
from src.models.reward import RewardFunction
from src.utils.visualization import TrainingVisualizer
from src.utils.callbacks import VisualizationCallback
import yaml

def main():
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
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
    with open("data/train/motion_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    # 初始化tokenizer和模型
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 创建数据集
    train_dataset = MotionDataset(train_data, model_name)
    
    # 初始化奖励函数
    reward_fn = RewardFunction(config, visualizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=10,
        save_steps=config["training"]["save_steps"],
        gradient_accumulation_steps=4,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # 设置训练器
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_function=reward_fn,
        callbacks=[VisualizationCallback(visualizer)]  # 添加可视化回调
    )
    
    # 开始训练
    trainer.train()
    
    # 生成训练分析报告
    visualizer.plot_rewards_history()
    visualizer.plot_reward_distribution()
    visualizer.plot_prediction_accuracy()
    visualizer.plot_learning_curves()
    training_report = visualizer.generate_training_report()
    
    # 保存训练状态
    visualizer.save_visualization_state()
    
    # 保存最终模型
    trainer.save_model("./final_model")
    
    # 关闭wandb
    wandb.finish()

if __name__ == "__main__":
    main()