"""
思维链 (Chain-of-Thought) Prompt 生成工具
"""

# 可以考虑在这里加入一些 Few-Shot 示例
# 例如，从 AQuA 数据集中挑选一些具有代表性的问题和推理过程
FEW_SHOT_EXAMPLES = """
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Options:
A) 24
B) 72
C) 48
D) 144
E) 96
Let's think step by step.
Natalia sold 48 clips in April.
In May, she sold half as many as in April, which is 48 / 2 = 24 clips.
Altogether, she sold 48 + 24 = 72 clips.
So the answer is B.

Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Options:
A) 50
B) 40
C) 70
D) 60
E) 80
Let's think step by step.
The sum of the 15 numbers is 15 * 40 = 600.
If 10 is added to each number, the new sum will be 600 + (15 * 10) = 600 + 150 = 750.
The new average will be 750 / 15 = 50.
So the answer is A.
"""

def create_aqua_cot_prompt(question_text):
    """为AQuA问题创建思维链提示.
    
    Args:
        question_text (str): 预处理后的问题文本 (包含问题和选项).
        
    Returns:
        str: 完整的 CoT prompt.
    """
    # 基本的 CoT 引导语
    # 您可以根据实验效果调整这里的引导语和Few-shot示例
    # prompt = f"{FEW_SHOT_EXAMPLES}\n\nQ: {question_text}\nLet's think step by step."
    prompt = f"Q: {question_text}\nLet's think step by step."
    return prompt

# 测试 (可选)
if __name__ == '__main__':
    sample_question = "What is the value of x in the equation 2x + 3 = 7?\nOptions:\nA) 1\nB) 2\nC) 3\nD) 4\nE) 5"
    prompt = create_aqua_cot_prompt(sample_question)
    print("Generated CoT Prompt:")
    print(prompt)