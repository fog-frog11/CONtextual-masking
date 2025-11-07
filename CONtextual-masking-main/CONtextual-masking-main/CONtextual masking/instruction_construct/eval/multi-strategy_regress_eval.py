import argparse
import json
import torch
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import math
import numpy as np
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import random


def smiles_to_fp(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits) if mol else None
    except: return None

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_regression_value_from_text(text):
    text = text.strip()
    try:
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match: return float(match.group(0))
        return None
    except: return None

def create_dynamic_regression_prompt(task_description, query_masked_smiles, value_range, few_shot_examples):
    min_val, max_val = value_range
    base_instruction = (
        f"You are an expert molecular chemist. Your task is to {task_description}.\n"
        f"The expected output is a single numerical value, typically between {min_val:.2f} and {max_val:.2f}. "
        "Your output MUST be only the number. Do NOT include units, text, or explanations.\n\n"
    )
    
    if not few_shot_examples:
        instruction = base_instruction + "Analyze the following molecule's structure. Note that a key part has been masked as '[MASK]'.\n\n"
        instruction += f"SMILES: {query_masked_smiles}\n"
    else:
        instruction = base_instruction + "Here are some examples based on the provided molecules:\n"
        for example_smiles, example_label in few_shot_examples:
            instruction += f"SMILES: {example_smiles}\nPredicted Value: {example_label}\n\n"
        instruction += "Now, based on these examples, predict the value for the following molecule:\n"
        instruction += f"SMILES: {query_masked_smiles}\n"
    instruction += "Predicted Value:"
    return instruction

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on a REGRESSION task with different few-shot strategies.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--task_desc', type=str, required=True)
    parser.add_argument('--few_shot_strategy', type=str, default='similarity', 
                        choices=['similarity', 'random'],
                        help="Strategy to select few-shot examples. 'diversity' is not applicable for regression.")
    return parser.parse_args()

def load_model(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer

def evaluate(model, tokenizer, test_data, train_data, device, num_shots, task_desc, few_shot_strategy):
    y_true, y_pred = [], []
    results = []
    
    print(f"Preprocessing training data for '{few_shot_strategy}' strategy...")
    train_samples_with_fp = []
    labels_for_range = []
    for sample in tqdm(train_data):
        try:
            labels_for_range.append(float(sample['output']))
            sample['fp'] = smiles_to_fp(sample.get('original_smiles', sample.get('input')))
            if sample['fp']:
                train_samples_with_fp.append(sample)
        except (ValueError, TypeError):
            continue
    
    value_range = (min(labels_for_range), max(labels_for_range)) if labels_for_range else (0,0)
    print(f"Value range for prompts determined from training data: {value_range[0]:.2f} to {value_range[1]:.2f}")

    for item in tqdm(test_data, desc=f"Evaluating with {few_shot_strategy} strategy"):
        query_smiles_original = item.get('original_smiles', item.get('input'))
        query_smiles_for_prompt = item['input']
        try:
            true_label = float(item['output'].strip())
        except ValueError: continue

        few_shot_examples = []
        if num_shots > 0:
            query_fp = smiles_to_fp(query_smiles_original)
            if query_fp:
                examples_raw = []
                if few_shot_strategy == 'similarity':
                    similarities = sorted([(tanimoto_similarity(query_fp, ts['fp']), ts) for ts in train_samples_with_fp], key=lambda x: x[0], reverse=True)
                    examples_raw = [s for _, s in similarities[:num_shots]]
                
                elif few_shot_strategy == 'random':
                    if len(train_samples_with_fp) >= num_shots:
                        examples_raw = random.sample(train_samples_with_fp, num_shots)

                if len(examples_raw) == num_shots:
                    few_shot_examples = [(ex.get('original_smiles', ex['input']), ex['output']) for ex in examples_raw]

        full_input = create_dynamic_regression_prompt(task_desc, query_smiles_for_prompt, value_range, few_shot_examples)
        
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        
        pred_value, gen_text = 0.0, ""
        try:
            generation_config = GenerationConfig(max_new_tokens=15, temperature=0.1, do_sample=False, num_beams=4, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            with torch.no_grad():
                generation_output = model.generate(**inputs, generation_config=generation_config)
            gen_ids = generation_output.sequences[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            parsed_value = get_regression_value_from_text(gen_text)
            if parsed_value is not None: pred_value = parsed_value
        except Exception: pass

        y_true.append(true_label)
        y_pred.append(pred_value)
        results.append({"instruction": full_input, "input": query_smiles_for_prompt, "output": item['output'], "model_output": gen_text, "predicted_value": pred_value})
        
    return y_true, y_pred, results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)
    
    with open(args.test_file, 'r', encoding='utf-8') as f: test_data = json.load(f)
    with open(args.train_file, 'r', encoding='utf-8') as f: train_data = json.load(f)

    y_true, y_pred, results = evaluate(model, tokenizer, test_data, train_data, device, args.num_shots, args.task_desc, args.few_shot_strategy)

    output_dir = os.path.dirname(args.output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(results, f, indent=2)

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n--- Results for Task: {os.path.basename(args.test_file)} (Strategy: {args.few_shot_strategy}) ---")
        print(f"RMSE = {rmse:.4f}")
        print(f"R-squared = {r2:.4f}")
        metrics = {"RMSE": round(rmse, 4), "R2": round(r2, 4)}
        metrics_file = args.output_file.replace(".json", f"_{args.few_shot_strategy}_metrics.json")
        with open(metrics_file, 'w') as f: json.dump(metrics, f, indent=2)
        print(f"Evaluation complete. Metrics saved to {metrics_file}")
    except ValueError as e:
        print(f"Error calculating RMSE/R2: {e}")

if __name__ == "__main__":
    main()