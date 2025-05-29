import yaml
import os
import json
from src.data.generator import MotionDataGenerator

def main():
    # 加载配置
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化数据生成器
    generator = MotionDataGenerator(config)
    
    # 生成数据集
    splits = {
        "train": config["data"]["train"]["n_samples"],
        "test_in": config["data"]["test_in"]["n_samples"],
        "test_out": config["data"]["test_out"]["n_samples"]
    }
    
    for split, n_samples in splits.items():
        dataset = generator.generate_dataset(split=split, n_samples=n_samples)
        
        # 确保目录存在
        os.makedirs(f"data/{split}", exist_ok=True)
        
        # 保存数据
        with open(f"data/{split}/motion_data.json", "w") as f:
            json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    main()