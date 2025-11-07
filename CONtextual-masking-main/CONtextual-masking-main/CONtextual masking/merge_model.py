
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True, help='')
    parser.add_argument('--lora_model_path', type=str, required=True, help='')
    parser.add_argument('--output_path', type=str, required=True, help='')
    return parser.parse_args()

def main():
    args = parse_args()


    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

   
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)

    
    merged_model = model.merge_and_unload()

    
    merged_model.save_pretrained(args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)

    

if __name__ == '__main__':
    main()
