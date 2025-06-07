import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.dataset import MotionDataset
import yaml
from tqdm import tqdm
import os
import re

class TestEvaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def process_batch(self, batch_data):
        """处理一个批次的数据"""
        # 获取提示和响应
        prompts = [item["query"] for item in batch_data]
        responses = [item["response"] for item in batch_data]
        
        # tokenize输入
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config["model"]["max_length"]
        )
        
        # 记录每个prompt的token长度
        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        
        return encoded, responses, prompt_lengths
        
    def evaluate_batch(self, batch_data):
        """评估单个批次"""
        # 处理输入数据
        encoded, responses, prompt_lengths = self.process_batch(batch_data)
        
        # 移动到设备
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config["model"]["max_length"],
                num_return_sequences=1,
                temperature=self.config["training"]["temperature"],
                top_p=self.config["training"]["top_p"],
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        processed_predictions = []
        # 对每个输出单独处理
        for output, prompt_length in zip(outputs, prompt_lengths):
            # 只取prompt_length之后的token
            completion_ids = output[prompt_length:]
            # 解码只包含回答部分的token
            raw_prediction = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            
            # 处理[End]标签和提取答案
            end_pos = raw_prediction.find('[End]')
            if end_pos != -1:
                raw_prediction = raw_prediction[:end_pos + len('[End]')]
                
            # 提取预测坐标
            answer = ""
            answer_match = re.search(r'\[ANSWER\](.*?)(?=\[End\]|$)', raw_prediction, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
                
            processed_predictions.append({
                "processed_output": raw_prediction,
                "predicted_coordinates": answer
            })
        
        return processed_predictions, responses
    
    def evaluate_dataset(self, data, split_name):
        """评估整个数据集"""
        all_predictions = []
        all_targets = []
        batch_size = self.config["training"]["per_device_train_batch_size"]
        
        # 创建批次
        for i in tqdm(range(0, len(data), batch_size), desc=f"评估 {split_name}"):
            batch_data = data[i:i + batch_size]
            predictions, targets = self.evaluate_batch(batch_data)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
        return {
            "predictions": all_predictions,
            "targets": all_targets,
        }


def main():
    # 加载配置
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 设置检查点路径
    checkpoint_path = "/root/autodl-tmp/final_model/checkpoint-60"
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"检查点路径不存在: {checkpoint_path}")
    
    print(f"正在加载检查点: {checkpoint_path}")
    
    # 加载模型和tokenizer
    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.float16 if config["training"].get("fp16", False) else torch.float32
    )
    
    # 设置tokenizer的特殊token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    print("模型加载完成")
    
    # 创建评估器
    evaluator = TestEvaluator(model, tokenizer, config)
    
    # 加载测试数据
    with open("data/test_in/motion_data.json", "r", encoding="utf-8") as f:
        test_in_data = json.load(f)
    with open("data/test_out/motion_data.json", "r", encoding="utf-8") as f:
        test_out_data = json.load(f)
    
    # 评估模型
    print("\n评估in-distribution数据...")
    in_dist_results = evaluator.evaluate_dataset(test_in_data, "in-distribution")
    
    print("\n评估out-of-distribution数据...")
    out_dist_results = evaluator.evaluate_dataset(test_out_data, "out-of-distribution")
    
    # 创建结果目录
    os.makedirs("results/test_in", exist_ok=True)
    os.makedirs("results/test_out", exist_ok=True)
    
    # 保存预测结果
    print("\n保存预测结果...")
    with open("results/test_in/predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": in_dist_results["predictions"],
            "targets": in_dist_results["targets"]
        }, f, ensure_ascii=False, indent=2)
    
    with open("results/test_out/predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "predictions": out_dist_results["predictions"],
            "targets": out_dist_results["targets"]
        }, f, ensure_ascii=False, indent=2)
    
    print("\n测试完成！结果已保存到 results/test_in 和 results/test_out 目录")


if __name__ == "__main__":
    main()