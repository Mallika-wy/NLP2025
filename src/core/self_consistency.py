"""
Self-Consistency 核心逻辑实现
"""
import re
from collections import Counter
import configs.config as config
from src.core.model import LLMUtils
from src.core.prompt import create_aqua_cot_prompt


class SelfConsistencyRunner:
    def __init__(self, llm_utils: LLMUtils):
        self.llm_utils = llm_utils

    def generate_reasoning_paths(self, processed_question):
        """为给定问题生成多条推理路径."""
        prompt = create_aqua_cot_prompt(processed_question)
        paths = []
        for i in range(config.experiment.num_reasoning_paths):
            print(f"  Generating path {i+1}/{config.experiment.num_reasoning_paths}...")
            generated_text = self.llm_utils.generate_text(
                prompt_text=prompt,
                temperature=config.experiment.temperature,
                top_k=config.experiment.top_k,
                max_new_tokens=config.experiment.max_new_tokens
            )
            paths.append(generated_text)
            # print(f"    Path {i+1} Raw Output: {generated_text[:200]}...") # 打印部分原始输出用于调试
        return paths

    def extract_answer_from_path(self, reasoning_path):
        """从单条推理路径中提取最终答案选项 (A, B, C, D, or E)."""
        # 这是一个关键步骤，需要根据模型输出的格式进行调整
        # 尝试匹配常见的答案声明模式，例如 "The answer is A.", "So the answer is (B)", "Final Answer: C"
        # 优先匹配末尾的答案
        match = re.search(r"[Tt]he answer is (?:\()?([A-E])(?:\))?\.?$", reasoning_path, re.MULTILINE)
        if match:
            return match.group(1).upper()

        # 备用匹配：寻找句子末尾的单个大写字母 A-E，可能被句号包围
        match = re.search(r"([A-E])\.?$", reasoning_path.strip(), re.MULTILINE)
        if match:
            return match.group(1).upper()
        
        # 更宽松的匹配，寻找文本中最后一个明确提及的选项字母
        # 例如 "Therefore, option A is correct."
        matches = re.findall(r"option ([A-E])", reasoning_path, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        # 如果上述方法都失败，可以尝试从末尾提取最后一个大写字母作为答案 (作为最后的手段)
        # 这可能不够鲁棒，需要根据实际输出进行优化
        potential_answers = re.findall(r"\b([A-E])\b", reasoning_path)
        if potential_answers:
            return potential_answers[-1].upper()

        print(f"Warning: Could not extract answer from path: {reasoning_path[:200]}...")
        return None  # 或返回一个特定的错误标识

    def get_final_answer_by_majority_vote(self, answers):
        """通过多数投票确定最终答案."""
        if not answers or all(a is None for a in answers):
            return None  # 没有有效答案
        
        valid_answers = [ans for ans in answers if ans is not None]
        if not valid_answers:
            return None
            
        vote_counts = Counter(valid_answers)
        # print(f"    Vote counts: {vote_counts}")
        most_common = vote_counts.most_common(1)
        return most_common[0][0] if most_common else None

    def run_single_question(self, processed_question):
        """对单个问题运行Self-Consistency流程."""
        reasoning_paths = self.generate_reasoning_paths(processed_question)
        
        extracted_answers = []
        for i, path in enumerate(reasoning_paths):
            answer = self.extract_answer_from_path(path)
            # print(f"    Path {i+1} Extracted Answer: {answer}")
            extracted_answers.append(answer)
            
        final_answer = self.get_final_answer_by_majority_vote(extracted_answers)
        
        return {
            "question": processed_question,
            "reasoning_paths": reasoning_paths,
            "extracted_answers": extracted_answers,
            "final_answer": final_answer
        }


# 测试 (可选)
if __name__ == "__main__":
    # 这个测试需要能成功加载模型
    print("Starting SelfConsistencyRunner test...")
    try:
        llm_utils_instance = LLMUtils()  # 这会尝试加载模型
        runner = SelfConsistencyRunner(llm_utils_instance)
        
        sample_aqua_question_processed = """If a train travels at 100 km/h for 3 hours, how far does it travel?
Options:
A) 200 km
B) 300 km
C) 400 km
D) 500 km
E) 100 km"""
        
        print(f"\nTesting with sample question: {sample_aqua_question_processed}")
        result = runner.run_single_question(sample_aqua_question_processed)
        
        print("\n--- Self-Consistency Result ---")
        print(f"Question: {result['question']}")
        print(f"Generated Paths ({len(result['reasoning_paths'])}):")
        for i, path in enumerate(result['reasoning_paths']):
            print(f"  Path {i+1}: {path[:150]}... (Extracted: {result['extracted_answers'][i]})")
        print(f"Final Answer by Majority Vote: {result['final_answer']}")

    except Exception as e:
        print(f"An error occurred during SelfConsistencyRunner test: {e}")
        print("Ensure your model paths in config.py are correct and you have enough resources.")
 