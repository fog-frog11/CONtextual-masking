import argparse
import pandas as pd
import json
from tokenizers import ByteLevelBPETokenizer
import random
from tqdm import tqdm
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# --- Helper Functions ---
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

def augment_smiles(smiles, num_augmentations=5):
    """Generates multiple equivalent SMILES strings for a given molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]
    
    augmented = {Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) for _ in range(num_augmentations * 2)}
    augmented.add(smiles)
    
    return list(augmented)[:num_augmentations]

def create_instruction(masked_smiles, task_description, few_shot_examples=None):
    """Creates an instruction prompt for a classification task."""
    base_instruction = (
        f"You are an expert molecule classifier. Your task is to {task_description}.\n"
        "Your output MUST be exactly 'Yes' or 'No'. Do NOT include any additional text, explanations, or examples after your answer.\n\n"
    )
    if few_shot_examples:
        instruction = "Task: Classification\n" + base_instruction + "Here are some examples based on molecules with similar structures:\n"
        for ex_smiles, ex_label in few_shot_examples:
            # 示例和查询都使用掩码SMILES
            instruction += f"SMILES: {ex_smiles}\nLabel: {ex_label}\n\n"
        instruction += "Now, based on these examples, classify the following molecule:\n"
        instruction += f"SMILES: {masked_smiles}\n"
    else:
        instruction = "Task: Classification\n" + base_instruction + "Analyze the following molecule's structure. Note that a key part of the molecule has been masked and is represented by '[MASK]'.\n\n"
        instruction += f"SMILES: {masked_smiles}\n"
    instruction += "[Classification Label]:"
    return instruction

def main():
    parser = argparse.ArgumentParser(
        description="Generate CLASSIFICATION instruction prompts with BPE masking, augmentation, and structure-aware few-shot examples.")
    
    # --- 参数定义 ---
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the trained BPE tokenizer's vocab.json file.")
    parser.add_argument('--smiles_col', type=str, default='smiles')
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=0, help="Number of few-shot examples. 0 for zero-shot.")
    parser.add_argument('--task_desc', type=str, required=True, help="A short description of the classification task.")
    parser.add_argument('--augment', action='store_true', help="Enable data augmentation for the minority class.")
    parser.add_argument('--max_ratio', type=int, default=5, help="Target max ratio of majority to minority class.")
    
    args = parser.parse_args()

    # --- 1. 加载分词器 ---
    print("Loading BPE tokenizer...")
    tokenizer_vocab_file = args.tokenizer_path
    tokenizer_merges_file = os.path.join(os.path.dirname(args.tokenizer_path), "merges.txt")
    if not os.path.exists(tokenizer_vocab_file) or not os.path.exists(tokenizer_merges_file):
        raise FileNotFoundError(f"Tokenizer files not found. Check path: {args.tokenizer_path}")
    tokenizer = ByteLevelBPETokenizer(vocab=tokenizer_vocab_file, merges=tokenizer_merges_file)
    if "[MASK]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(["[MASK]"])
    print("Tokenizer loaded successfully.")

    # --- 2. 加载和清洗数据 ---
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Initial data size: {len(df)}")
    
    original_size = len(df)
    df.dropna(subset=[args.label_col, args.smiles_col], inplace=True)
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors='coerce')
    df.dropna(subset=[args.label_col], inplace=True)
    df[args.label_col] = df[args.label_col].astype(int)
    
    cleaned_size = len(df)
    if original_size > cleaned_size:
        print(f"Data Cleaning: Removed {original_size - cleaned_size} rows with missing or invalid data.")
    print(f"Data size after cleaning: {cleaned_size}")
    if cleaned_size == 0:
        print(f"❌ Error: No valid data left.")
        return

    # --- 3. [可选] 数据增强 ---
    if args.augment:
        print("\n--- Data Augmentation Enabled ---")
        label_counts = df[args.label_col].value_counts()
        if len(label_counts) < 2:
            print("Warning: Only one class present. Augmentation skipped.")
        else:
            print(f"Initial class distribution:\n{label_counts}")
            majority_class, minority_class = label_counts.idxmax(), label_counts.idxmin()
            majority_count, minority_count = label_counts.max(), label_counts.min()

            target_minority_count = majority_count // args.max_ratio
            if minority_count > 0 and minority_count < target_minority_count:
                num_to_augment = target_minority_count - minority_count
                print(f"Augmenting minority class '{minority_class}' with {num_to_augment} new samples.")
                
                minority_df = df[df[args.label_col] == minority_class]
                augmented_rows = []
                
                while len(augmented_rows) < num_to_augment:
                    sample_to_augment = minority_df.sample(1).iloc[0]
                    original_smiles = str(sample_to_augment[args.smiles_col])
                    new_smiles_list = augment_smiles(original_smiles, num_augmentations=10)
                    
                    for new_smiles in new_smiles_list:
                        if new_smiles != original_smiles and len(augmented_rows) < num_to_augment:
                            new_row = sample_to_augment.to_dict()
                            new_row[args.smiles_col] = new_smiles
                            new_row['is_augmented'] = True
                            augmented_rows.append(new_row)
                
                if augmented_rows:
                    augmented_df = pd.DataFrame(augmented_rows)
                    df = pd.concat([df, augmented_df], ignore_index=True)
                    print(f"Data augmented. New total size: {len(df)}")
                    print(f"New class distribution:\n{df[args.label_col].value_counts()}")

    all_samples = df.to_dict('records')

    # --- 4. 预处理指纹 ---
    if args.num_shots > 0:
        print("Preprocessing dataset to compute fingerprints...")
        for sample in tqdm(all_samples):
            sample['fp'] = smiles_to_fp(str(sample[args.smiles_col]))
        all_samples = [s for s in all_samples if s.get('fp') is not None]

    # --- 5. 生成指令 ---
    instruction_data = []
    for i, query_sample in enumerate(tqdm(all_samples, desc=f"Generating {args.num_shots}-shot Prompts")):
        smiles = str(query_sample[args.smiles_col])
        label = "Yes" if query_sample[args.label_col] == 1 else "No"
        masked_smiles, masked_group = get_masked_smiles_bpe(smiles, tokenizer)
        
        few_shot_examples = None
        if args.num_shots > 0:
            query_fp = query_sample.get('fp')
            if not query_fp: continue
            
            other_samples = all_samples[:i] + all_samples[i+1:]
            similarities = sorted(
                [(tanimoto_similarity(query_fp, s['fp']), s) for s in other_samples if s.get('fp')],
                key=lambda x: x[0], reverse=True
            )
            top_k_examples = similarities[:args.num_shots]
            if len(top_k_examples) < args.num_shots: continue
            
            few_shot_examples = []
            for _, ex in top_k_examples:
                # !!! 关键修改：对示例也进行掩码 !!!
                ex_smiles, _ = get_masked_smiles_bpe(str(ex[args.smiles_col]), tokenizer)
                ex_label = "Yes" if ex[args.label_col] == 1 else "No"
                few_shot_examples.append((ex_smiles, ex_label))

        instruction = create_instruction(masked_smiles, args.task_desc, few_shot_examples)
        instruction_data.append({"instruction": instruction, "input": masked_smiles, "output": label, "original_smiles": smiles, "masked_group": masked_group})

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, indent=2)
    print(f"✅ Generated {len(instruction_data)} prompts and saved to {args.output_json}")

if __name__ == "__main__":
    main()