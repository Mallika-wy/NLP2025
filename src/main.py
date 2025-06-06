"""
主程序入口
"""
import torch
from src.core.model import LLMUtils
from src.core.self_consistency import SelfConsistencyRunner
from src.utils import logger
from configs.config import config

def main():
    """主程序入口"""
    try:
        # 初始化模型和tokenizer
        logger.info("正在初始化模型...")
        llm_utils = LLMUtils()
        
        # 创建Self-Consistency运行器
        logger.info("正在创建Self-Consistency运行器...")
        runner = SelfConsistencyRunner(llm_utils.model, llm_utils.tokenizer)
        
        # 运行批处理推理
        logger.info("开始运行批处理推理...")
        results = runner.run()
        
        # 计算准确率
        correct_count = sum(1 for r in results if r['predicted'] == r['correct'])
        accuracy = correct_count / len(results)
        
        # 输出结果统计
        logger.info("推理完成！")
        logger.info(f"总问题数: {len(results)}")
        logger.info(f"正确数量: {correct_count}")
        logger.info(f"准确率: {accuracy:.2%}")
        
        # 保存详细结果
        import json
        with open(config.paths.results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': accuracy,
                'total_questions': len(results),
                'correct_count': correct_count,
                'detailed_results': results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {config.paths.results_file}")
        
    except Exception as e:
        logger.error(f"运行时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 