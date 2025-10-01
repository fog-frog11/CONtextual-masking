import argparse
import pandas as pd
import json
import random
from tqdm import tqdm
import os
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tokenizers import ByteLevelBPETokenizer
from sklearn.model_selection import train_test_split

# --- Helper Function 1: SMILES Augmentation for Balancing ---
def augment_smiles(smiles, num_augmentations=5):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [smiles]
    augmented = {Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) for _ in range(num_augmentations * 2)}
    augmented.add(smiles)
    return list(augmented)[:num_augmentations]

# --- Helper Function 2: BPE Masking ---
def get_masked_smiles_bpe(smiles, tokenizer, mask_token="[MASK]"):
    encoding = tokenizer.encode(smiles)
    token_ids, tokens = encoding.ids, encoding.tokens
    if len(token_ids) <= 2: return smiles, "Too_Short_To_Mask"
    try:
        mask_token_id = tokenizer.token_to_id(mask_token)
        if mask_token_id is None: raise ValueError()
    except: return smiles, "Mask_Token_Not_Found"
    mask_idx = random.randint(0, len(token_ids) - 1)
    original_token = tokens[mask_idx]
    masked_token_ids = token_ids[:mask_idx] + [mask_token_id] + token_ids[mask_idx + 1:]
    masked_smiles = tokenizer.decode(masked_token_ids)
    return masked_smiles, f"BPE_Token:_{original_token}"

# --- Helper Function 3: Fingerprint Calculation ---
def smiles_to_fp(smiles, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits) if mol else None
    except: return None

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)
    
# --- Helper Function 4: Scaffold Split ---
def generate_scaffold(smiles, include_chirality=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
    except: return None

def scaffold_split(df, smiles_col, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-9
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        if scaffold: scaffolds[scaffold].append(i)
        else: scaffolds[smiles].append(i)
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_idx, val_idx, test_idx = [], [], []
    n_total = len(df)
    for group in scaffold_sets:
        if len(val_idx) / n_total < frac_valid: val_idx.extend(group)
        elif len(test_idx) / n_total < frac_test: test_idx.extend(group)
        else: train_idx.extend(group)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

# --- Helper Function 5: Instruction Formatting ---
def create_instruction(masked_smiles, task_description, few_shot_examples=None):
    base_instruction = (
        f"You are an expert molecule classifier. Your task is to {task_description}.\n"
        "Your output MUST be exactly 'Yes' or 'No'. Do NOT include any additional text, explanations, or examples after your answer.\n\n"
    )
    if few_shot_examples:
        instruction = base_instruction + "Here are some examples based on molecules with similar structures:\n"
        for ex_smiles, ex_label in few_shot_examples:
            instruction += f"SMILES: {ex_smiles}\nLabel: {ex_label}\n\n"
        instruction += "Now, based on these examples, classify the following molecule:\n"
        instruction += f"SMILES: {masked_smiles}\n"
    else:
        instruction = base_instruction + "Analyze the following molecule's structure. Note that a key part of the molecule has been masked and is represented by '[MASK]'.\n\n"
        instruction += f"SMILES: {masked_smiles}\n"
    instruction += "Label:"
    return instruction

def main():
    parser = argparse.ArgumentParser(description="Generate, deduplicate, scaffold-split, balance, and format instruction prompts for classification tasks.")
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the BPE tokenizer's vocab.json file.")
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--smiles_col', type=str, required=True)
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--task_desc', type=str, required=True)
    parser.add_argument('--num_shots', type=int, default=0)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--imbalance_ratio', type=int, default=5)
    
    args = parser.parse_args()

    # --- 1. 加载和清洗 ---
    print(f"\n--- Processing Task: {args.task_name.upper()} ---")
    df = pd.read_csv(args.input_csv)
    df.dropna(subset=[args.smiles_col, args.label_col], inplace=True)
    # 确保标签列是数值类型，无法转换的行将被丢弃
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors='coerce')
    df.dropna(subset=[args.label_col], inplace=True)
    df[args.label_col] = df[args.label_col].astype(int)
    
    # --- 2. 去重 ---
    print("Deduplicating molecules...")
    df['canonical_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True) if Chem.MolFromSmiles(smi) else None for smi in tqdm(df[args.smiles_col], desc="Canonicalizing")]
    df.dropna(subset=['canonical_smiles'], inplace=True)
    df.drop_duplicates(subset=['canonical_smiles'], keep='first', inplace=True)
    df = df.drop(columns=['canonical_smiles']).reset_index(drop=True)
    print(f"Data size after deduplication: {len(df)}")
    if len(df) < 20: # 增加一个检查，如果数据太少就警告并退出
        print(f"Warning: Dataset for task '{args.task_name}' is too small ({len(df)} samples) to proceed. Skipping.")
        return

    # --- 3. Scaffold Split ---
    print("Performing Scaffold Split (8:1:1)...")
    train_df, val_df, test_df = scaffold_split(df, smiles_col=args.smiles_col)
    print(f"Split sizes -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # --- 4. [可选] 增强训练集 ---
    if args.augment:
        print("\n--- Augmenting Training Set ---")
        label_counts = train_df[args.label_col].value_counts()
        if len(label_counts) < 2:
            print("Training set has only one class. Augmentation skipped.")
        else:
            majority_class, minority_class = label_counts.idxmax(), label_counts.idxmin()
            majority_count, minority_count = label_counts.max(), label_counts.min()
            if minority_count > 0 and majority_count / minority_count > args.imbalance_ratio:
                target_minority_count = max(1, majority_count // args.imbalance_ratio)
                num_to_augment = target_minority_count - minority_count
                
                print(f"Initial train distribution: {dict(label_counts)}")
                if num_to_augment > 0:
                    print(f"Augmenting minority class '{minority_class}' with {num_to_augment} new samples...")
                    minority_samples_df = train_df[train_df[args.label_col] == minority_class]
                    augmented_rows = []
                    while len(augmented_rows) < num_to_augment:
                        sample_to_augment = minority_samples_df.sample(1).iloc[0].to_dict()
                        new_smiles_list = augment_smiles(str(sample_to_augment[args.smiles_col]), 10)
                        for new_smiles in new_smiles_list:
                            if new_smiles != sample_to_augment[args.smiles_col] and len(augmented_rows) < num_to_augment:
                                new_row = sample_to_augment.copy()
                                new_row[args.smiles_col] = new_smiles
                                new_row['is_augmented'] = True
                                augmented_rows.append(new_row)
                    
                    if augmented_rows:
                        train_df = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
                        print(f"Augmentation complete. New training set size: {len(train_df)}")
                        print(f"New training class distribution:\n{train_df[args.label_col].value_counts()}")

    # --- 5. 生成指令 ---
    print("\n--- Generating Instruction Files ---")
    
    # !!! 关键修正：使用正确的构造函数加载Tokenizer !!!
    tokenizer_vocab_file = args.tokenizer_path
    tokenizer_merges_file = os.path.join(os.path.dirname(args.tokenizer_path), "merges.txt")
    if not os.path.exists(tokenizer_vocab_file) or not os.path.exists(tokenizer_merges_file):
        raise FileNotFoundError(f"Tokenizer files not found. Check path: {args.tokenizer_path}")
    tokenizer = ByteLevelBPETokenizer(vocab=tokenizer_vocab_file, merges=tokenizer_merges_file)
    if "[MASK]" not in tokenizer.get_vocab(): tokenizer.add_special_tokens(["[MASK]"])

    train_samples_with_fp = []
    if args.num_shots > 0:
        print("Preprocessing training set for few-shot example retrieval...")
        for r in train_df.to_dict('records'):
            fp = smiles_to_fp(str(r[args.smiles_col]))
            if fp: r['fp'] = fp; train_samples_with_fp.append(r)

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        if len(split_df) == 0:
            print(f"Skipping {split_name} set as it is empty.")
            continue
            
        instruction_data = []
        print(f"Processing {split_name} set ({len(split_df)} samples)...")
        for _, row in tqdm(split_df.iterrows()):
            smiles = str(row[args.smiles_col])
            label = "Yes" if row[args.label_col] == 1 else "No"
            masked_smiles, _ = get_masked_smiles_bpe(smiles, tokenizer)
            
            few_shot_examples = None
            if args.num_shots > 0:
                query_fp = smiles_to_fp(smiles)
                if query_fp and train_samples_with_fp:
                    sims = sorted([(tanimoto_similarity(query_fp, s['fp']), s) for s in train_samples_with_fp if 'fp' in s], key=lambda x: x[0], reverse=True)
                    examples_raw = [s for _, s in sims[:args.num_shots]]
                    few_shot_examples = [(get_masked_smiles_bpe(str(ex[args.smiles_col]), tokenizer)[0], "Yes" if ex[args.label_col] == 1 else "No") for ex in examples_raw]

            instruction = create_instruction(masked_smiles, args.task_desc, few_shot_examples)
            instruction_data.append({"instruction": instruction, "input": masked_smiles, "output": label, "original_smiles": smiles})
        
        output_path = os.path.join(args.output_dir, f"{args.task_name}_{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(instruction_data, f, indent=2)
        print(f"Saved {len(instruction_data)} samples to {output_path}")

if __name__ == "__main__":
    main()