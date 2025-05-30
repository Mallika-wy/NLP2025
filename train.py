import os
import json
import wandb
from trl import GRPOTrainer, GRPOConfig
from src.data.dataset import MotionDataset
from src.models.reward import accuracy_reward, format_reward, cot_reward, trend_reward
from src.utils.visualization import TrainingVisualizer
from src.utils.callbacks import VisualizationCallback
import yaml

# 设置wandb, 离线模式
os.environ["WANDB_API_KEY"] = "dd6af3f5f51014cd49b6f4ec4e4943d917b46ed6"
os.environ["WANDB_MODE"] = "offline"

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
    
    # model_name
    model_name = config["model"]["name"]

    # 创建数据集
    train_dataset = MotionDataset(train_data)  # 移除 tokenizer_name 参数
    
    # 设置奖励函数
    reward_funcs = [
        accuracy_reward,
        format_reward,
        cot_reward,
        trend_reward
    ]
    
    # 设置训练参数
    grpo_config = GRPOConfig(
        # 基础训练设置
        output_dir="./results",
        learning_rate=1e-6,            # 从小的学习率开始
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=True,
        gradient_checkpointing=True,
        
        # GRPO特定参数
        beta=0.04,                     # KL系数，控制与参考模型的差异
        epsilon=0.2,                   # 裁剪范围
        num_iterations=1,              # 每个批次的迭代次数
        loss_type="bnpo",             # 使用bnpo损失函数
        scale_rewards=True,           # 对奖励进行标准化
        reward_weights=config["reward_weights"],  # 从配置文件中读取奖励权重
        
        # 生成参数
        max_prompt_length=384,        # 最大提示长度
        max_completion_length=128,    # 最大生成长度
        num_generations=4,            # 每个提示生成8个样本
        temperature=0.7,             # 稍微降低温度以增加确定性
        top_p=0.9,                   # 使用nucleus sampling
        
        # 日志和可视化
        logging_steps=10,            # 每10步记录一次日志
        save_steps=100,             # 每100步保存一次模型
        log_completions=True,       # 记录生成样本
        wandb_log_unique_prompts=True,
        
        # 其他优化设置
        disable_dropout=True,       # 训练时禁用dropout
        mask_truncated_completions=True,  # 处理截断的生成结果
    )

    # 设置训练器
    trainer = GRPOTrainer(
        model=model_name,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        callbacks=[VisualizationCallback(visualizer)]
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