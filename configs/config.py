"""
实验配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型相关配置"""
    # model_path: str = "meta-llama/Llama-2-7b-hf"
    # tokenizer_path: str = "meta-llama/Llama-2-7b-hf"
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"  # "cuda" or "cpu"

@dataclass
class DataConfig:
    """数据集相关配置"""
    dataset_path: str = "data/test.json"
    max_samples: Optional[int] = 200  # None表示使用全部样本

@dataclass
class ExperimentConfig:
    """实验参数配置"""
    num_reasoning_paths: int = 5  # 生成的推理路径数量
    temperature: float = 0.7      # 采样温度
    top_k: int = 50              # Top-K 采样
    max_new_tokens: int = 512    # 每条路径最大生成长度
    random_seed: int = 42        # 随机种子

@dataclass
class PathConfig:
    """路径相关配置"""
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir: str = os.path.join(base_dir, "results")
    data_dir: str = os.path.join(base_dir, "data")
    
    def __post_init__(self):
        """确保必要的目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    @property
    def log_file(self) -> str:
        return os.path.join(self.results_dir, "experiment_log.txt")
    
    @property
    def results_file(self) -> str:
        return os.path.join(self.results_dir, "results.json")
    
    @property
    def dataset_path(self) -> str:
        return os.path.join(self.data_dir, "test.json")

class Config:
    """配置类，整合所有配置"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.paths = PathConfig()
        
        # 使用PathConfig中的实际路径更新DataConfig
        self.data.dataset_path = self.paths.dataset_path

# 创建全局配置实例
config = Config()

if __name__ == "__main__":
    # 测试配置
    print(f"Model path: {config.model.model_path}")
    print(f"Results directory: {config.paths.results_dir}")
    print(f"Dataset path: {config.data.dataset_path}")
    print(f"Log file: {config.paths.log_file}") 