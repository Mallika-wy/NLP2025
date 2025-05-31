import json
import swanlab
from swanlab.integration.transformers import SwanLabCallback
from trl import GRPOTrainer, GRPOConfig
from src.data.dataset import MotionDataset
from src.models.reward import accuracy_reward, cot_reward, format_reward
from src.utils.visualization import TrainingVisualizer
from src.utils.callbacks import VisualizationCallback
import yaml


def main():
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
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
    ]
    
    # 设置训练参数
    grpo_config = GRPOConfig(
        # 基础训练设置
        output_dir="./results",
        run_name="qwen-motion-training",
        learning_rate=1e-6,            # 从小的学习率开始
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        # gradient_checkpointing=True,    
        # gradient_checkpointing_kwargs= {
        #     "use_reentrant": True 
        # },
        
        # GRPO特定参数
        beta=0.04,                     # KL系数，控制与参考模型的差异
        epsilon=0.2,                   # 裁剪范围
        num_iterations=1,              # 每个批次的迭代次数
        loss_type="bnpo",             # 使用bnpo损失函数
        scale_rewards=True,           # 对奖励进行标准化
        reward_weights=config["reward_weights"],  # 从配置文件中读取奖励权重
        
        # 生成参数
        max_prompt_length=400,        # 最大提示长度
        max_completion_length=200,    # 最大生成长度
        num_generations=4,            # 每个提示生成8个样本
        temperature=0.7,             # 稍微降低温度以增加确定性
        top_p=0.9,                   # 使用nucleus sampling
        
        # 日志和可视化
        logging_steps=10,
        save_steps=100,
        log_completions=True,
        report_to="none",  # 禁用默认的wandb报告
        
        # 其他优化设置
        disable_dropout=True,       # 训练时禁用dropout
        mask_truncated_completions=False,  # 处理截断的生成结果
    )

    # 实例化SwanLabCallback
    swanlab_callback = SwanLabCallback(
        project=config["project"], 
        experiment_name=config["experiment_name"]
    )

    # 设置训练器
    trainer = GRPOTrainer(
        model=model_name,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        callbacks=[swanlab_callback]
    )

    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("./final_model")


if __name__ == "__main__":
    main()