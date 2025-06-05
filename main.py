"""
实验主运行脚本
"""
import time
import random
import torch

from configs.config import config
from src.core import LLMUtils, SelfConsistencyRunner
from src.data import load_aqua_dataset, preprocess_aqua_question, get_correct_answer_letter
from src.utils import logger, Evaluator

def run_experiment():
    """运行完整的Self-Consistency实验."""
    logger.info("Starting Self-Consistency experiment...")
    logger.info(f"Using model: {config.model.model_path}")
    logger.info(f"Using dataset: {config.data.dataset_path}")
    logger.info(f"Number of reasoning paths: {config.experiment.num_reasoning_paths}")
    logger.info(f"Temperature: {config.experiment.temperature}, Top-K: {config.experiment.top_k}")

    # 设置随机种子保证可复现性
    random.seed(config.experiment.random_seed)
    torch.manual_seed(config.experiment.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.experiment.random_seed)

    try:
        # 1. 初始化模型工具
        logger.info("Initializing LLM utilities...")
        llm_utils = LLMUtils()
        if not llm_utils.model or not llm_utils.tokenizer:
            logger.error("Failed to load model or tokenizer. Exiting.")
            return

        # 2. 加载数据集
        logger.info("Loading AQuA dataset...")
        dataset = load_aqua_dataset()
        if not dataset:
            logger.error("Failed to load dataset or dataset is empty. Exiting.")
            return
        logger.info(f"Loaded {len(dataset)} questions from AQuA dataset.")

        # 3. 初始化 Self-Consistency 运行器和评估器
        sc_runner = SelfConsistencyRunner(llm_utils)
        evaluator = Evaluator()

        # 4. 遍历数据集中的每个问题
        start_time = time.time()
        for i, item in enumerate(dataset):
            question_number = i + 1
            logger.info(f"\nProcessing question {question_number}/{len(dataset)}: {item.get('question')[:100]}...")
            
            processed_question_text = preprocess_aqua_question(item)
            correct_answer_letter = get_correct_answer_letter(item)

            if not correct_answer_letter:
                logger.warning(f"Question {question_number} does not have a correct answer specified. Skipping.")
                continue
            
            try:
                # 运行Self-Consistency
                sc_result = sc_runner.run_single_question(processed_question_text)
                
                # 记录结果
                evaluator.record_single_result(item, sc_result, correct_answer_letter)
                logger.info(f"Question {question_number}: SC Predicted: {sc_result['final_answer']}, Correct: {correct_answer_letter}, Match: {sc_result['final_answer'] == correct_answer_letter}")
            except Exception as e:
                logger.error(f"Error processing question {question_number}: {item.get('question')}. Error: {e}", exc_info=True)
                error_sc_result = {
                    'question': processed_question_text,
                    'reasoning_paths': ['Error during generation'],
                    'extracted_answers': [None],
                    'final_answer': None
                }
                evaluator.record_single_result(item, error_sc_result, correct_answer_letter)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\n--- Experiment Finished ---")
        logger.info(f"Total questions processed: {evaluator.total_count}")
        logger.info(f"Correct predictions: {evaluator.correct_count}")
        logger.info(f"Accuracy: {evaluator.get_accuracy():.2f}%")
        logger.info(f"Total execution time: {total_time:.2f} seconds.")
        logger.info(f"Average time per question: {(total_time / evaluator.total_count if evaluator.total_count > 0 else 0):.2f} seconds.")

        # 5. 保存所有结果
        evaluator.save_results()
        logger.info(f"Results and logs saved to: {config.paths.results_dir}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the experiment: {e}", exc_info=True)

if __name__ == "__main__":
    run_experiment() 