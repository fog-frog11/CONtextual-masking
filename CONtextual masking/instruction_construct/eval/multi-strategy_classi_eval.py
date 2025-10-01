import argparse
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
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
    try: mol = Chem.MolFromSmiles(smiles); return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits) if mol else None
    except: return None

def tanimoto_similarity(fp1, fp2): return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_prediction_from_text(text):
    text = text.strip().lower()
    if text.startswith("yes"): return 1
    if text.startswith("no"): return 0
    return None

def create_dynamic_few_shot_prompt(task_description, query_masked_smiles, few_shot_examples):
    base_instruction = (f"You are an expert molecule classifier. Your task is to {task_description}.\n"
                      "Your output MUST be exactly 'Yes' or 'No'. Do NOT include any additional text or explanations.\n\n")
    if not few_shot_examples:
        return base_instruction + f"Analyze the following molecule's structure. Note that a key part has been masked as '[MASK]'.\n\nSMILES: {query_masked_smiles}\nLabel:"
    else:
        instruction = base_instruction + "Here are some examples based on the provided molecules:\n"
        for example_smiles, example_label in few_shot_examples:
            instruction += f"SMILES: {example_smiles}\nLabel: {example_label}\n\n"
        instruction += f"Now, based on these examples, classify the following molecule:\nSMILES: {query_masked_smiles}\nLabel:"
        return instruction


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model with different few-shot strategies.")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--task_desc', type=str, required=True)
    parser.add_argument('--few_shot_strategy', type=str, default='similarity', choices=['similarity', 'diversity', 'random'])
    return parser.parse_args()

def load_model(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    model.eval()
    return model, tokenizer

def evaluate(model, tokenizer, test_data, train_data, device, num_shots, task_desc, few_shot_strategy):
    y_true, y_pred, y_prob, results = [], [], [], []
    
    try:
        yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        no_token_id = tokenizer("No", add_special_tokens=False).input_ids[0]
    except Exception as e:
        print(f"❌ Critical Error: Could not get token IDs. {e}"); return [], [], [], []
    
    print(f"Preprocessing training data for '{few_shot_strategy}' strategy...")
    all_train_samples_with_fp = [s for s in train_data if 'fp' not in s or s['fp'] is None]
    for sample in tqdm(all_train_samples_with_fp):
        sample['fp'] = smiles_to_fp(sample.get('original_smiles', sample.get('input')))
    all_train_samples_with_fp = [s for s in all_train_samples_with_fp if s.get('fp')]

    for item in tqdm(test_data, desc=f"Evaluating with {few_shot_strategy} strategy"):
        query_smiles_original = item.get('original_smiles', item.get('input'))
        query_smiles_for_prompt = item['input']
        true_label = 1 if item['output'].strip().lower() == 'yes' else 0
        few_shot_examples = []
        
        if num_shots > 0:
            query_fp = smiles_to_fp(query_smiles_original)
            if query_fp:
                
                CANDIDATE_POOL_SIZE = 50 
                similarities = sorted([(tanimoto_similarity(query_fp, ts['fp']), ts) for ts in all_train_samples_with_fp], key=lambda x: x[0], reverse=True)
                candidate_pool = [s for _, s in similarities[:CANDIDATE_POOL_SIZE]]
                
                examples_raw = []
                if few_shot_strategy == 'similarity':
                    examples_raw = candidate_pool[:num_shots]
                
                elif few_shot_strategy == 'diversity':
                    pos_in_pool = [s for s in candidate_pool if s['output'].lower() == 'yes']
                    neg_in_pool = [s for s in candidate_pool if s['output'].lower() == 'no']
                    num_pos = math.ceil(num_shots / 2)
                    num_neg = num_shots // 2
                    pos_ex = random.sample(pos_in_pool, min(num_pos, len(pos_in_pool)))
                    neg_ex = random.sample(neg_in_pool, min(num_neg, len(neg_in_pool)))
                    examples_raw = pos_ex + neg_ex
                    random.shuffle(examples_raw)

                elif few_shot_strategy == 'random':
                    if len(candidate_pool) >= num_shots:
                        examples_raw = random.sample(candidate_pool, num_shots)

                if len(examples_raw) >= num_shots:
                    examples_raw = examples_raw[:num_shots]
                    few_shot_examples = [(ex.get('original_smiles', ex['input']), ex['output']) for ex in examples_raw]

        full_input = create_dynamic_few_shot_prompt(task_desc, query_smiles_for_prompt, few_shot_examples)
        
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(device)
        pred_label, yes_prob_value, gen_text = 0, 0.5, ""
        try:
            generation_config = GenerationConfig(max_new_tokens=10, temperature=0.1, do_sample=False, num_beams=4, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            with torch.no_grad():
                generation_output = model.generate(**inputs, generation_config=generation_config, return_dict_in_generate=True, output_scores=True)
            gen_ids = generation_output.sequences[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            parsed_pred = get_prediction_from_text(gen_text)
            if parsed_pred is not None:
                pred_label, yes_prob_value = parsed_pred, float(parsed_pred)
            elif generation_output.scores:
                last_token_logits = generation_output.scores[-1][0]
                probs = F.softmax(last_token_logits, dim=-1)
                yes_prob_value = (probs[yes_token_id].item() + 1e-9) / (probs[yes_id].item() + probs[no_id].item() + 1e-9)
                pred_label = 1 if yes_prob_value >= 0.5 else 0
        except Exception: pass
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_prob.append(yes_prob_value)
        results.append({"instruction": full_input, "input": query_smiles_for_prompt, "output": item['output'], "model_output": gen_text, "predicted_label": pred_label, "yes_prob": yes_prob_value})
    
    return y_true, y_pred, y_prob, results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)
    
    with open(args.test_file, 'r', encoding='utf-8') as f: test_data = json.load(f)
    with open(args.train_file, 'r', encoding='utf-8') as f: train_data = json.load(f)

    y_true, y_pred, y_prob, results = evaluate(model, tokenizer, test_data, train_data, device, args.num_shots, args.task_desc, args.few_shot_strategy)

    output_dir = os.path.dirname(args.output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(results, f, indent=2)

    task_name = os.path.basename(args.test_file)
    metrics = {}
    try:
        print(f"\n--- Results for Task: {task_name} (Strategy: {args.few_shot_strategy}) ---")
        acc = accuracy_score(y_true, y_pred)
        metrics["ACC"] = round(acc, 4)
        print(f"Accuracy = {acc:.4f}")
        if 'low' in task_name.lower():
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            prc_auc = auc(recall, precision)
            metrics["PRC-AUC"] = round(prc_auc, 4)
            print(f"PRC-AUC = {prc_auc:.4f}")
        else:
            roc_auc = roc_auc_score(y_true, y_prob)
            metrics["ROC-AUC"] = round(roc_auc, 4)
            print(f"ROC-AUC = {roc_auc:.4f}")
        metrics_file = args.output_file.replace(".json", f"_{args.few_shot_strategy}_metrics.json")
        with open(metrics_file, "w") as f: json.dump(metrics, f, indent=2)
        print(f"Evaluation complete. Metrics saved to {metrics_file}")
    except ValueError as e:
        print(f"❌ Error calculating metrics: {e}")

if __name__ == "__main__":
    main()