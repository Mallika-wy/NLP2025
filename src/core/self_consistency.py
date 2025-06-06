"""
Self-Consistency 核心逻辑实现
"""
import re
from collections import Counter
import configs.config as config
from src.core.model import LLMUtils
from src.core.prompt import create_aqua_cot_prompt
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from src.utils import logger
from src.data.AQuADataset import create_dataloader


class SelfConsistencyRunner:
    """Self-Consistency推理实现"""
    
    def __init__(self, model, tokenizer):
        """
        初始化Self-Consistency运行器
        
        Args:
            model: 语言模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.model.device

    def generate_reasoning_paths(self, prompts: List[str]) -> List[List[str]]:
        """
        为一批问题并行生成多条推理路径
        """
        batch_size = len(prompts)
        num_paths = config.experiment.num_reasoning_paths
        
        # 将每个prompt重复num_paths次，这样可以一次性生成所有路径
        expanded_prompts = [p for p in prompts for _ in range(num_paths)]
        
        # 一次性对所有prompt进行编码
        inputs = self.tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        input_lengths = [len(self.tokenizer.encode(p)) for p in expanded_prompts]
        
        # 一次性生成所有路径
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.experiment.max_new_tokens,
                temperature=config.experiment.temperature,
                top_k=config.experiment.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1  # 每个输入生成一个序列
            )
        
        # 解码所有生成的文本
        all_generations = []
        for output, input_length in zip(outputs, input_lengths):
            response = self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True
            ).strip()
            all_generations.append(response)

        # 重组结果，使每个问题的所有路径在一起
        paths = []
        for i in range(0, len(all_generations), num_paths):
            paths.append(all_generations[i:i + num_paths])
        
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

    def run(self) -> List[Dict[str, Any]]:
        """
        运行Self-Consistency推理
        
        Returns:
            推理结果列表
        """
        # 创建数据加载器
        dataloader = create_dataloader(
            config.data.dataset_path,
            config.experiment.batch_size,
            num_workers=config.experiment.num_workers
        )
        
        results = []
        
        # 批处理推理
        for batch in tqdm(dataloader, desc="Processing batches"):
            # 为每个问题创建提示
            prompts = [
                create_aqua_cot_prompt(f"{q}\nOptions:\n" + "\n".join(opts))
                for q, opts in zip(batch['questions'], batch['options'])
            ]
            
            # 生成推理路径
            batch_paths = self.generate_reasoning_paths(prompts)
            
            # 处理每个问题的结果
            for i, (paths, question, options, correct) in enumerate(zip(
                batch_paths, batch['questions'], batch['options'], batch['corrects']
            )):
                # 统计每个选项的投票
                votes = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
                for path in paths:
                    # 这里需要实现从推理路径中提取答案的逻辑
                    answer = self.extract_answer_from_path(path)
                    if answer in votes:
                        votes[answer] += 1
                
                # 获取得票最多的选项
                predicted = max(votes.items(), key=lambda x: x[1])[0]
                
                # 记录结果
                results.append({
                    'question': question,
                    'options': options,
                    'correct': correct,
                    'predicted': predicted,
                    'votes': votes,
                    'paths': paths
                })
                
                # 记录日志
                logger.info(f"Question {len(results)}: Predicted={predicted}, Correct={correct}")
        
        return results

    def extract_answer(self, reasoning_path: str) -> str:
        """
        从推理路径中提取答案
        
        Args:
            reasoning_path: 推理路径文本
            
        Returns:
            提取的答案（A-E）
        """
        # 这里需要实现具体的答案提取逻辑
        # 可以使用正则表达式或其他方法
        # 示例实现：
        if "answer is A" in reasoning_path.lower():
            return "A"
        elif "answer is B" in reasoning_path.lower():
            return "B"
        elif "answer is C" in reasoning_path.lower():
            return "C"
        elif "answer is D" in reasoning_path.lower():
            return "D"
        elif "answer is E" in reasoning_path.lower():
            return "E"
        else:
            # 如果没有找到明确的答案，返回默认值
            return "A"


# 测试 (可选)
if __name__ == "__main__":
    # 这个测试需要能成功加载模型
    print("Starting SelfConsistencyRunner test...")
    try:
        llm_utils_instance = LLMUtils()  # 这会尝试加载模型
        runner = SelfConsistencyRunner(llm_utils_instance.model, llm_utils_instance.tokenizer)
        
        sample_aqua_question_processed = """If a train travels at 100 km/h for 3 hours, how far does it travel?
Options:
A) 200 km
B) 300 km
C) 400 km
D) 500 km
E) 100 km"""
        
        print(f"\nTesting with sample question: {sample_aqua_question_processed}")
        result = runner.run()
        
        print("\n--- Self-Consistency Result ---")
        for i, result in enumerate(result):
            print(f"Question {i+1}:")
            print(f"  Question: {result['question']}")
            print(f"  Options: {', '.join(result['options'])}")
            print(f"  Correct Answer: {result['correct']}")
            print(f"  Predicted Answer: {result['predicted']}")
            print(f"  Votes: {result['votes']}")
            print(f"  Paths: {', '.join(result['paths'])}")

    except Exception as e:
        print(f"An error occurred during SelfConsistencyRunner test: {e}")
        print("Ensure your model paths in config.py are correct and you have enough resources.")
 