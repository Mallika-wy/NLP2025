"""
模型加载和推理工具
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import configs.config as config

class LLMUtils:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """加载预训练模型和分词器"""
        try:
            print(f"Loading model from {config.model.model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model.model_path,
                torch_dtype=torch.float16, # 使用float16以减少显存占用
                device_map="auto" # 自动分配到可用GPU
            )
            print(f"Loading tokenizer from {config.model.tokenizer_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.tokenizer_path,
                padding_side='left'
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            # 可以在这里添加更详细的错误处理或日志记录
            raise

    def generate_text(self, prompt_text, temperature, top_k, max_new_tokens):
        """使用指定参数生成文本"""
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer not loaded.")
            return None

        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(config.model.device)
        
        # 生成文本
        with torch.no_grad(): # 关闭梯度计算，减少显存占用和计算开销
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True, # 启用采样
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的token IDs
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 通常我们只对新生成的token解码，避免重复prompt
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

# 测试 (可选)
if __name__ == '__main__':
    try:
        llm_utils = LLMUtils()
        if llm_utils.model and llm_utils.tokenizer:
            test_prompt = "Translate the following English text to French: 'Hello, how are you?'"
            generated_output = llm_utils.generate_text(
                prompt_text=test_prompt,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K,
                max_new_tokens=50
            )
            print(f"\nTest Prompt: {test_prompt}")
            print(f"Generated Text: {generated_output}")
    except Exception as e:
        print(f"An error occurred during testing: {e}") 