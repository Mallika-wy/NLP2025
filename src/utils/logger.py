"""
通用工具函数
"""
import logging
import datetime
import os
import configs.config as config

def setup_logger():
    """配置日志记录器."""
    #确保日志文件目录存在
    log_dir = os.path.dirname(config.paths.log_file)
    os.makedirs(log_dir, exist_ok=True)
        
    # 1. 创建logger实例
    logger = logging.getLogger('SelfConsistencyExperiment')  # 创建一个名为'SelfConsistencyExperiment'的logger
    logger.setLevel(logging.INFO)  # 设置最低日志级别为INFO
    
    # 2. 检查是否已有处理器，避免重复添加
    if not logger.handlers:
        # 3. 创建文件处理器（将日志写入文件）
        fh = logging.FileHandler(config.paths.log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 4. 创建控制台处理器（将日志输出到控制台）
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 5. 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 包含：时间 - logger名称 - 日志级别 - 日志消息
        
        # 6. 将格式应用到处理器
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 7. 将处理器添加到logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

# 初始化并获取logger实例，以便在其他模块中直接导入使用
logger = setup_logger()

# 测试 (可选)
if __name__ == '__main__':
    logger.info("This is an info message from utils.py.")
    logger.warning("This is a warning message from utils.py.")
    print(f"Log file should be at: {config.paths.log_file}") 