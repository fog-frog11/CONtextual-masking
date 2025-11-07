import argparse
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import math
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_fp(smiles, radius=2, nBits=2048):
    """Converts SMILES to Morgan Fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    except:
        return None

def tanimoto_similarity(fp1, fp2):
    """Calculates Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test JSON file.")
   
    parser.add_argument('--train_file', type=str, help="Path to the train JSON file for building few-shot examples.")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=3, help="Number of few-shot examples to use during evaluation.")
    
    return parser.parse_args()

def load_model(model_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return model, tokenizer

def get_logits_yes_no(output_text):
    text = output_text.strip().lower()
    if text.startswith("yes"):
        return 1
    elif text.startswith("no"):
        return 0
    else:
        return None


def create_dynamic_few_shot_prompt(query_smiles, few_shot_examples):
    instruction = (
        "Task: Classification\n"
        "You are an expert molecule classifier. Your task is to determine its ability to inhibit HIV-1 integrase.\n"
        "Your output MUST be exactly 'Yes' or 'No'. Do NOT include any additional text, explanations, or examples after your answer.\n\n"
        "Here are some examples based on similar molecules:\n"
    )
    for example_smiles, example_label in few_shot_examples:
        
        instruction += f"[SMILES] {example_smiles}\n[Label] {example_label}\n\n"
    
    instruction += "Now, based on these examples, classify the following molecule:\n"
    instruction += f"[SMILES] {query_smiles}\n"
    instruction += "[Classification Label]:"
    return instruction

def evaluate(model, tokenizer, test_data, train_data, device, num_shots):
    y_true, y_pred, y_prob, results = [], [], [], []
    invalid_count = 0
    
    try:
        yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
        no_token_id = tokenizer("No", add_special_tokens=False).input_ids[0]
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.eos_token_id is None: tokenizer.eos_token_id = tokenizer.pad_token_id
    except Exception as e:
        print(f" Error getting token IDs: {e}")
        return [], [], [], []

    
    train_samples_with_fp = []
    if num_shots > 0 and train_data:
        print("Preprocessing training data for few-shot example retrieval...")
        for sample in tqdm(train_data):
            
            smiles = sample.get('original_smiles') or sample.get('input')
            fp = smiles_to_fp(smiles)
            if fp:
                train_samples_with_fp.append({
                    'fp': fp,
                    # 'input' 
                    'input': sample['input'], 
                    'output': sample['output']
                })
    
    
    for item in tqdm(test_data, desc="Evaluating"):
        
        query_smiles_original = item.get('original_smiles') or item.get('input')
        query_smiles_for_prompt = item['input'] 
        
        label_text = item['output'].strip()
        true_label = 1 if label_text.lower() == 'yes' else 0

        
        full_input = item['instruction'] 
        if num_shots > 0 and train_samples_with_fp:
            query_fp = smiles_to_fp(query_smiles_original)
            if query_fp:
                
                similarities = []
                for train_sample in train_samples_with_fp:
                    sim = tanimoto_similarity(query_fp, train_sample['fp'])
                    similarities.append((sim, train_sample))
                
                
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_k_examples = similarities[:num_shots]
                
                few_shot_examples = [(ex['input'], ex['output']) for _, ex in top_k_examples]
                
                
                full_input = create_dynamic_few_shot_prompt(query_smiles_for_prompt, few_shot_examples)
            
        
        inputs = tokenizer(full_input, return_tensors="pt", truncation=True, padding=True).to(device)
        
        
        pred_label = None
        yes_prob = None
        gen_text = "" 
        try:
            generation_config = GenerationConfig(
                max_new_tokens=10,
                temperature=0.4,   
                do_sample=True,       
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                
            )
            generation_output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True 
            )
            gen_ids = generation_output.sequences[0][inputs["input_ids"].shape[-1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if gen_text: 
                pred_label = get_logits_yes_no(gen_text)
                if pred_label is not None:
                    yes_prob = 1.0 if pred_label == 1 else 0.0
                else: # Fallback
                    try:
                        if generation_output.scores:
                            last_token_logits = generation_output.scores[-1][0]
                            probs_all = F.softmax(last_token_logits, dim=-1)
                            if 0 <= yes_token_id < probs_all.shape[-1]:
                                yes_prob = probs_all[yes_token_id].item()
                                pred_label = 1 if yes_prob >= 0.5 else 0
                            else:
                                pred_label, yes_prob = 0, 0.5
                        else:
                            pred_label, yes_prob = 0, 0.5
                    except Exception as e:
                        print(f"Fallback error: {e}")
                        pred_label, yes_prob = 0, 0.5
            else: # Empty output, fallback
                invalid_count += 1
                try:
                    if generation_output.scores:
                        last_token_logits = generation_output.scores[-1][0]
                        probs_all = F.softmax(last_token_logits, dim=-1)
                        if 0 <= yes_token_id < probs_all.shape[-1]:
                            yes_prob = probs_all[yes_token_id].item()
                            pred_label = 1 if yes_prob >= 0.5 else 0
                        else:
                           pred_label, yes_prob = 0, 0.5
                    else:
                        pred_label, yes_prob = 0, 0.5
                except Exception as e:
                    print(f"Fallback error (empty): {e}")
                    pred_label, yes_prob = 0, 0.5

        except Exception as e:
            print(f"Generation failed: {e}")
            pred_label, yes_prob = 0, 0.5
            invalid_count += 1

        if pred_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(yes_prob if yes_prob is not None else 0.5)

        results.append({
            "instruction": full_input, 
            "input": query_smiles_for_prompt,
            "output": label_text,
            "model_output": gen_text,
            "predicted_label": pred_label,
            "yes_prob": yes_prob 
        })
    
    print(f"Empty/failed outputs: {invalid_count}/{len(test_data)}")
    return y_true, y_pred, y_prob, results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)

    try:
        with open(args.test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        
        train_data = None
        if args.train_file:
            with open(args.train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            print(f"Loaded {len(train_data)} samples from train file for few-shot examples.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON - {e}")
        return

    
    y_true, y_pred, y_prob, results = evaluate(model, tokenizer, test_data, train_data, device, args.num_shots)

    
    valid_samples = [(yt, yp, pr) for yt, yp, pr in zip(y_true, y_pred, y_prob) if yt is not None and pr is not None and math.isfinite(pr)]
    if not valid_samples:
        print("No valid predictions to compute metrics.")
        return

    y_true_clean, y_pred_clean, y_prob_clean = zip(*valid_samples)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f" Evaluation results saved to: {args.output_file}")

    try:
        auc = roc_auc_score(y_true_clean, y_prob_clean)
        acc = accuracy_score(y_true_clean, y_pred_clean)
        print(f"AUC = {auc:.4f}, ACC = {acc:.4f}")
        metrics = { "AUC": round(auc, 4), "ACC": round(acc, 4) }
        metrics_file = args.output_file.replace(".json", "_metrics.json")
        with open(metrics_file, "w") as f: json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
    except ValueError as e:
        print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    main()