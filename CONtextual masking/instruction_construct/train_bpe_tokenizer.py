import argparse
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
import os
import glob


def train_bpe_from_parquet(input_path, smiles_col, output_dir, vocab_size=5000, sample_frac=1.0):
    """
    Trains a BPE tokenizer from SMILES strings in Parquet files.
    The input_path can be a single .parquet file or a directory of .parquet files.
    """
    print(f"Reading data from: {input_path}")

    
    if os.path.isdir(input_path):
        
        parquet_files = glob.glob(os.path.join(input_path, '*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in directory: {input_path}")
        print(f"Found {len(parquet_files)} Parquet files. Reading them into a single DataFrame...")
        df = pd.read_parquet(input_path)
    elif os.path.isfile(input_path) and input_path.endswith('.parquet'):
        print("Reading a single Parquet file...")
        df = pd.read_parquet(input_path)
    else:
        raise ValueError("Input path must be a .parquet file or a directory containing .parquet files.")

    print(f"Successfully loaded {len(df)} rows.")

    
    if smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in the data. Available columns: {df.columns.tolist()}")

    
    if sample_frac < 1.0:
        print(f"Sampling {sample_frac * 100}% of the data...")
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Using {len(df)} samples for training the tokenizer.")

    smiles_list = df[smiles_col].dropna().tolist()

    
    text_file = "smiles_corpus_temp.txt"
    print(f"Writing {len(smiles_list)} SMILES to a temporary file for training...")
    with open(text_file, "w", encoding='utf-8') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

    
    print("Training the BPE tokenizer... (this may take a while for large datasets)")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_file],
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "[MASK]"]
    )

    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

    
    os.remove(text_file)
    print(f"BPE tokenizer trained and saved to {output_dir}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for SMILES from Parquet files.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input Parquet file or directory.")
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help="Name of the SMILES column in the Parquet files.")
    parser.add_argument('--output_dir', type=str, default='./zinc20_bpe_tokenizer',
                        help="Directory to save the tokenizer files.")
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help="Vocabulary size for the BPE tokenizer (recommend a larger size for large datasets).")
    parser.add_argument('--sample_frac', type=float, default=1.0,
                        help="Fraction of the dataset to use for training (e.g., 0.1 for 10%). Default is 1.0 (all data).")

    args = parser.parse_args()

    
    try:
        import pyarrow
    except ImportError:
        print("PyArrow not found. Please install it first: pip install pyarrow")
        return

    train_bpe_from_parquet(args.input_path, args.smiles_col, args.output_dir, args.vocab_size, args.sample_frac)


if __name__ == "__main__":
    main()