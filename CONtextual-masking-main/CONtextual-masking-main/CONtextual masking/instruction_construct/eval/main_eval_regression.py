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
    instruction = base_instruction + "Here are some examples based on molecules with similar structures:\n"
    for example_smiles, example_label in few_shot_examples:
        instruction += f"SMILES: {example_smiles}\nPredicted Value: {example_label}\n\n"
    instruction += "Now, based on these examples, predict the value for the following molecule:\n"
    instruction += f"SMILES: {query_masked_smiles}\nPredicted Value:"
    return instruction

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on a REGRESSION task.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, help="Path to the train JSON file for building few-shot examples and determining value range.")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--task_desc', type=str, required=True)
    return parser.parse_args()

def load_model(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer

def evaluate(model, tokenizer, test_data, train_data, device, num_shots, task_desc):
    y_true, y_pred = [], []
    results = []
    
    train_samples_with_fp = []
    value_range = (0.0, 0.0)

    if train_data:
        print("Preprocessing training data...")
        labels_for_range = [float(s['output']) for s in train_data if s.get('output')]
        if labels_for_range:
            value_range = (min(labels_for_range), max(labels_for_range))
            print(f"Value range for prompts determined from training data: {value_range[0]:.2f} to {value_range[1]:.2f}")

        if num_shots > 0:
            for sample in tqdm(train_data, desc="Caching train fingerprints"):
                smiles = sample.get('original_smiles', sample.get('input'))
                fp = smiles_to_fp(smiles)
                if fp:
                    train_samples_with_fp.append({'fp': fp, 'original_smiles': smiles, 'output': sample['output']})

    for item in tqdm(test_data, desc=f"Evaluating {os.path.basename(args.test_file)}"):
        query_smiles_original = item.get('original_smiles', item.get('input'))
        query_smiles_for_prompt = item['input']
        
        try:
            true_label = float(item['output'].strip())
        except ValueError: continue

        full_input = item['instruction']
        if num_shots > 0 and train_samples_with_fp:
            query_fp = smiles_to_fp(query_smiles_original)
            if query_fp:
                similarities = sorted([(tanimoto_similarity(query_fp, ts['fp']), ts) for ts in train_samples_with_fp], key=lambda x: x[0], reverse=True)
                top_k_examples = similarities[:num_shots]
                if len(top_k_examples) == num_shots:
                    few_shot_examples = [(ex['original_smiles'], ex['output']) for _, ex in top_k_examples]
                    full_input = create_dynamic_regression_prompt(task_desc, query_smiles_for_prompt, value_range, few_shot_examples)
        
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        
        pred_value, gen_text = 0.0, ""
        try:
            generation_config = GenerationConfig(max_new_tokens=15, temperature=0.1, do_sample=False, num_beams=4, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            with torch.no_grad():
                
                generation_output = model.generate(
                    **inputs, 
                    generation_config=generation_config,
                    return_dict_in_generate=True 
                )
            gen_ids = generation_output.sequences[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            parsed_value = get_regression_value_from_text(gen_text)
            if parsed_value is not None: pred_value = parsed_value

        except Exception as e:
            print(f"Warning: Generation failed for a sample. Error: {e}")

        y_true.append(true_label)
        y_pred.append(pred_value)
        results.append({"instruction": full_input, "input": query_smiles_for_prompt, "output": item['output'], "model_output": gen_text, "predicted_value": pred_value})
        
    return y_true, y_pred, results

def main():
    global args
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)
    
    with open(args.test_file, 'r', encoding='utf-8') as f: test_data = json.load(f)
    train_data = None
    if args.train_file:
        with open(args.train_file, 'r', encoding='utf-8') as f: train_data = json.load(f)
    else:
        print("Warning: --train_file not provided. Value range for prompts will be based on test data, which is not ideal.")
        train_data = test_data

    y_true, y_pred, results = evaluate(model, tokenizer, test_data, train_data, device, args.num_shots, args.task_desc)

    output_dir = os.path.dirname(args.output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(results, f, indent=2)

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n--- Task: {os.path.basename(args.test_file)} ---")
        print(f"RMSE = {rmse:.4f}")
        print(f"R-squared = {r2:.4f}")
        metrics = {"RMSE": round(rmse, 4), "R2": round(r2, 4)}
        metrics_file = args.output_file.replace(".json", "_metrics.json")
        with open(metrics_file, 'w') as f: json.dump(metrics, f, indent=2)
        print(f"Evaluation complete. Metrics saved to {metrics_file}")
    except ValueError as e:
        print(f"Error calculating RMSE: {e}")

if __name__ == "__main__":
    main()