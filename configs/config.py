"""
实验配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """模型相关配置"""
    # model_path: str = "meta-llama/Llama-2-7b-hf"
    # tokenizer_path: str = "meta-llama/Llama-2-7b-hf"
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    """数据集相关配置"""
    dataset_path: str = "data/test.json"
    max_samples: Optional[int] = 200  # None表示使用全部样本

@dataclass
class ExperimentConfig:
    """实验参数配置"""
    # 批处理参数
    batch_size: int = 8  # 批处理大小
    num_workers: int = 0  # 数据加载器工作进程数
    
    # 生成参数
    num_reasoning_paths: int = 5  # 每个问题生成的推理路径数
    temperature: float = 0.7  # 生成温度
    top_k: int = 50  # top-k采样参数
    max_new_tokens: int = 512  # 最大生成token数
    
    # 其他参数
    seed: int = 42  # 随机种子
    use_few_shot: bool = False  # 是否使用少样本示例

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