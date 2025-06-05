"""
实验评估和结果记录工具
"""
import json
import os
import datetime
import configs.config as config

def ensure_results_dir_exists():
    """确保结果目录存在."""
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

class Evaluator:
    def __init__(self):
        ensure_results_dir_exists()
        self.all_results = []
        self.correct_count = 0
        self.total_count = 0

    def record_single_result(self, question_data, sc_result, correct_answer_letter):
        """
        记录单个问题的处理结果。
        
        Args:
            question_data (dict): 原始的AQuA问题条目。
            sc_result (dict): SelfConsistencyRunner返回的结果。
            correct_answer_letter (str): 该问题的正确答案字母 (A, B, C, D, or E)。
        """
        self.total_count += 1
        is_correct = False
        if sc_result['final_answer'] and sc_result['final_answer'] == correct_answer_letter:
            self.correct_count += 1
            is_correct = True
        
        result_entry = {
            'original_question': question_data.get('question'),
            'options': question_data.get('options'),
            'processed_question': sc_result['question'],
            'reasoning_paths': sc_result['reasoning_paths'],
            'extracted_answers_from_paths': sc_result['extracted_answers'],
            'final_answer_by_sc': sc_result['final_answer'],
            'correct_answer_letter': correct_answer_letter,
            'is_correct_by_sc': is_correct
        }
        self.all_results.append(result_entry)
        return result_entry # 返回记录的条目，方便打印或即时查看

    def get_accuracy(self):
        """计算当前准确率."""
        if self.total_count == 0:
            return 0.0
        return (self.correct_count / self.total_count) * 100

    def save_results(self):
        """将所有结果和总结保存到文件."""
        ensure_results_dir_exists()
        summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_path': config.MODEL_PATH,
            'dataset_path': config.DATASET_PATH,
            'num_reasoning_paths': config.NUM_REASONING_PATHS,
            'temperature': config.TEMPERATURE,
            'top_k': config.TOP_K,
            'max_new_tokens': config.MAX_NEW_TOKENS,
            'max_samples_tested': config.MAX_SAMPLES if config.MAX_SAMPLES is not None else 'all',
            'total_questions': self.total_count,
            'correct_predictions': self.correct_count,
            'accuracy': self.get_accuracy(),
        }
        
        results_data = {
            'summary': summary,
            'detailed_results': self.all_results
        }
        
        try:
            with open(config.RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {config.RESULTS_FILE}")
        except Exception as e:
            print(f"Error saving results to {config.RESULTS_FILE}: {e}")
            
        # 也可以将摘要信息打印到日志文件
        try:
            with open(config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write("\n--- Experiment Summary ---\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("-------------------------\n")
            print(f"Summary also appended to {config.LOG_FILE}")
        except Exception as e:
            print(f"Error writing summary to log file {config.LOG_FILE}: {e}")

# 测试 (可选)
if __name__ == '__main__':
    evaluator = Evaluator()
    
    # 模拟一些结果
    sample_q_data1 = {'question': 'Q1', 'options': ['A) X', 'B) Y'], 'correct': 'A'}
    sample_sc_res1 = {'question': 'Q1 full', 'reasoning_paths': ['Path1 for Q1'], 'extracted_answers': ['A'], 'final_answer': 'A'}
    evaluator.record_single_result(sample_q_data1, sample_sc_res1, 'A')

    sample_q_data2 = {'question': 'Q2', 'options': ['A) P', 'B) Q'], 'correct': 'B'}
    sample_sc_res2 = {'question': 'Q2 full', 'reasoning_paths': ['Path1 for Q2', 'Path2 for Q2'], 'extracted_answers': ['A', 'B'], 'final_answer': 'B'}
    evaluator.record_single_result(sample_q_data2, sample_sc_res2, 'B')
    
    sample_q_data3 = {'question': 'Q3', 'options': ['A) M', 'B) N'], 'correct': 'A'}
    sample_sc_res3 = {'question': 'Q3 full', 'reasoning_paths': ['Path1 for Q3 (N)'], 'extracted_answers': ['N'], 'final_answer': 'N'}
    evaluator.record_single_result(sample_q_data3, sample_sc_res3, 'A')

    print(f"\nAccuracy: {evaluator.get_accuracy():.2f}%")
    evaluator.save_results()
    print(f"Check {config.RESULTS_DIR} for output files.") 