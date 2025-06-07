import os
import sys
import json
import swanlab
import torch
from swanlab.integration.transformers import SwanLabCallback
from trl import GRPOTrainer, GRPOConfig
from src.data.dataset import MotionDataset
from src.models.reward import accuracy_reward, format_reward
import yaml
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_checkpoint(trainer, output_dir):
    """安全地保存检查点"""
    try:
        # 创建临时目录
        temp_dir = os.path.join(output_dir, "temp_checkpoint")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存模型
        trainer.model.save_pretrained(
            temp_dir,
            safe_serialization=True  # 使用safetensors
        )
        
        # 保存tokenizer
        trainer.tokenizer.save_pretrained(temp_dir)
        
        # 保存训练参数
        trainer.save_state()
        
        # 如果临时保存成功，移动到最终位置
        final_dir = os.path.join(output_dir, f"checkpoint-{trainer.state.global_step}")
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(temp_dir, final_dir)
        
        print(f"成功保存检查点到 {final_dir}")
        
    except Exception as e:
        print(f"保存检查点时发生错误: {str(e)}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False
    
    return True

def main():
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载数据
    with open("data/train/motion_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    # 设置模型和分词器
    model_name = config["model"]["name"]

    # 创建数据集
    train_dataset = MotionDataset(train_data)
    
    # 设置奖励函数
    reward_funcs = [
        accuracy_reward,
        format_reward,
    ]

    train_config = config["training"]
    
    # 设置训练参数
    grpo_config = GRPOConfig(
        # 基础训练设置
        output_dir=train_config["output_dir"],
        run_name=train_config["run_name"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        fp16=train_config["fp16"],
        
        # 内存优化设置
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        
        # GRPO特定参数
        beta=train_config["beta"],
        epsilon=train_config["epsilon"],
        num_iterations=train_config["num_iterations"],
        loss_type=train_config["loss_type"],
        scale_rewards=train_config["scale_rewards"],
        reward_weights=train_config["reward_weights"],
        
        # 生成参数
        max_prompt_length=train_config["max_prompt_length"],
        max_completion_length=train_config["max_completion_length"],
        num_generations=train_config["num_generations"],
        temperature=train_config["temperature"],
        top_p=train_config["top_p"],
        
        # 日志和可视化
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        log_completions=train_config["log_completions"],
        report_to=train_config["report_to"],
        
        # 其他优化设置
        disable_dropout=train_config["disable_dropout"],
        mask_truncated_completions=train_config["mask_truncated_completions"],
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
    try:
        print("start training")
        trainer.train()
    except Exception as e:
        if "unexpected pos" in str(e):
            print("检查点保存出错，尝试使用安全保存方式...")
            save_checkpoint(trainer, trainer.args.output_dir)
        else:
            raise e
    
    # 保存最终模型
    final_save_success = save_checkpoint(trainer, "/root/autodl-tmp/final_model")
    if not final_save_success:
        print("警告：最终模型保存失败")


if __name__ == "__main__":
    main()