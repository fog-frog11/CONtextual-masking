# merge_lora_with_base.py

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True, help='路径或模型名，例如 meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--lora_model_path', type=str, required=True, help='LoRA adapter 的保存路径')
    parser.add_argument('--output_path', type=str, required=True, help='融合后保存的位置')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"加载基础模型: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"加载LoRA adapter: {args.lora_model_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)

    print("融合权重中...")
    merged_model = model.merge_and_unload()

    print(f"保存融合模型到: {args.output_path}")
    merged_model.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)

    print("权重融合完成 ✅")

if __name__ == '__main__':
    main()
