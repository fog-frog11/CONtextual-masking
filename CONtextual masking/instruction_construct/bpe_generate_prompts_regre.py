import argparse
import pandas as pd
import json
from tokenizers import ByteLevelBPETokenizer
import random
from tqdm import tqdm
import os
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_masked_smiles_bpe(smiles, tokenizer, mask_token="[MASK]"):
    """Masks a SMILES string by masking a random BPE token."""
    encoding = tokenizer.encode(smiles)
    token_ids, tokens = encoding.ids, encoding.tokens
    if len(token_ids) <= 2:
        return smiles, "Too_Short_To_Mask"
    try:
        mask_token_id = tokenizer.token_to_id(mask_token)
        if mask_token_id is None: raise ValueError(f"'{mask_token}' not in tokenizer's vocabulary.")
    except Exception as e:
        print(f"Warning: Error getting mask token ID: {e}")
        return smiles, "Mask_Token_Not_Found"
        
    mask_idx = random.randint(0, len(token_ids) - 1)
    original_token = tokens[mask_idx]
    masked_token_ids = token_ids[:mask_idx] + [mask_token_id] + token_ids[mask_idx + 1:]
    masked_smiles = tokenizer.decode(masked_token_ids)
    return masked_smiles, f"BPE_Token:_{original_token}"


def smiles_to_fp(smiles, radius=2, nBits=2048):
    """Converts SMILES to Morgan Fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits) if mol else None
    except:
        return None

def tanimoto_similarity(fp1, fp2):
    """Calculates Tanimoto similarity."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def create_regression_instruction(masked_smiles, task_description, value_range, few_shot_examples=None):
    """Creates an instruction prompt for regression tasks, including the expected value range."""
    min_val, max_val = value_range
    
    base_instruction = (
        f"You are an expert molecular chemist. Your task is to {task_description}.\n"
        f"The expected output is a single numerical value, typically between {min_val:.2f} and {max_val:.2f}. "
        "Your output MUST be only the number. Do NOT include units, text, or explanations.\n\n"
    )
    
    if few_shot_examples:
        instruction = base_instruction + "Here are some examples based on molecules with similar structures:\n"
        for ex_smiles, ex_label in few_shot_examples:
            instruction += f"SMILES: {ex_smiles}\nPredicted Value: {ex_label}\n\n"
        instruction += "Now, based on these examples, predict the value for the following molecule:\n"
        instruction += f"SMILES: {masked_smiles}\n"
    else: # Zero-shot
        instruction = base_instruction + "Analyze the following molecule's structure. Note that a key part has been masked as '[MASK]'.\n\n"
        instruction += f"SMILES: {masked_smiles}\n"

    instruction += "Predicted Value:"
    return instruction


def main():
    parser = argparse.ArgumentParser(description="Generate instruction prompts for REGRESSION tasks with BPE masking and value range constraints.")
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the trained BPE tokenizer's vocab.json file.")
    parser.add_argument('--smiles_col', type=str, default='smiles')
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--task_desc', type=str, required=True)
    
    args = parser.parse_args()

    
    print("Loading BPE tokenizer...")
    tokenizer_vocab_file = args.tokenizer_path
    merges_path = os.path.join(os.path.dirname(args.tokenizer_path), "merges.txt")
    if not os.path.exists(tokenizer_vocab_file) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found. Check path: {args.tokenizer_path}")

    
    tokenizer = ByteLevelBPETokenizer(
        vocab=tokenizer_vocab_file,
        merges=merges_path,
    )
    if "[MASK]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(["[MASK]"])
    print("Tokenizer loaded successfully.")
    
    
    df = pd.read_csv(args.input_csv)
    df.dropna(subset=[args.label_col, args.smiles_col], inplace=True)
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors='coerce')
    df.dropna(subset=[args.label_col], inplace=True)
    print(f"Loaded and cleaned {len(df)} samples.")
    if len(df) == 0: return

    
    min_value = df[args.label_col].min()
    max_value = df[args.label_col].max()
    value_range = (min_value, max_value)
    print(f"Value range for task '{args.task_desc}' detected: {value_range[0]:.2f} to {value_range[1]:.2f}")

    all_samples = df.to_dict('records')
    
    
    if args.num_shots > 0:
        print("Preprocessing dataset to compute fingerprints...")
        for sample in tqdm(all_samples):
            sample['fp'] = smiles_to_fp(str(sample[args.smiles_col]))
        all_samples = [s for s in all_samples if s.get('fp') is not None]

    
    instruction_data = []
    for i, query_sample in enumerate(tqdm(all_samples, desc=f"Generating {args.num_shots}-shot Prompts")):
        smiles = str(query_sample[args.smiles_col])
        label = f"{float(query_sample[args.label_col]):.4f}"
        masked_smiles, _ = get_masked_smiles_bpe(smiles, tokenizer)

        few_shot_examples = None
        if args.num_shots > 0:
            query_fp = query_sample.get('fp')
            if not query_fp: continue
            
            other_samples = [s for j, s in enumerate(all_samples) if i != j]
            similarities = sorted(
                [(tanimoto_similarity(query_fp, s['fp']), s) for s in other_samples if s.get('fp')],
                key=lambda x: x[0], reverse=True
            )
            top_k_examples = similarities[:args.num_shots]
            if len(top_k_examples) < args.num_shots: continue

            few_shot_examples = []
            for _, ex in top_k_examples:
                ex_smiles = str(ex[args.smiles_col])
                ex_label = f"{float(ex[args.label_col]):.4f}"
                few_shot_examples.append((ex_smiles, ex_label))
            
        instruction = create_regression_instruction(masked_smiles, args.task_desc, value_range, few_shot_examples)

        instruction_data.append({
            "instruction": instruction,
            "input": masked_smiles,
            "output": label,
            "original_smiles": smiles
        })

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2)

    print(f"Generated {len(instruction_data)} regression prompts and saved to {args.output_json}")

if __name__ == "__main__":
    main()

