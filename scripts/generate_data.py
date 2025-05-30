import os
import sys
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import yaml
import json
from src.data.generator import MotionDataGenerator

def main():
    # 加载配置
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
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