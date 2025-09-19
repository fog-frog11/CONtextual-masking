import argparse
import json
import random
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_json_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a JSON dataset into training, validation, and test sets.
    """
    
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    print(f"Reading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    if len(data) > 1:
        random.shuffle(data)

    
    try:
        train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42)
    except ValueError:
         
        print("Warning: Dataset too small to split. Placing all data in the training set.")
        return data, [], []

    
    if len(temp_data) < 2:
        return train_data, temp_data, []

    
    remaining_ratio = val_ratio + test_ratio
    val_size_in_temp = val_ratio / remaining_ratio
    
    try:
        val_data, test_data = train_test_split(temp_data, train_size=val_size_in_temp, random_state=42)
    except ValueError:
        print("Warning: Remaining data too small to split into validation and test. Placing all remaining in validation.")
        return train_data, temp_data, []


    return train_data, val_data, test_data

def save_to_json(data, output_path):
    """Saves a list of dictionaries to a JSON file."""
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Split a JSON dataset into named train, val, and test files.")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split files.")
    
    parser.add_argument('--task_name', type=str, required=True, help="Name of the task to prefix output files (e.g., 'bace', 'hiv').")
    
    args = parser.parse_args()

    
    train_data, val_data, test_data = split_dataset(args.input_json)
    
    print(f"\nSplitting complete for task '{args.task_name}':")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    
    
    train_output_path = os.path.join(args.output_dir, f'{args.task_name}_train.json')
    save_to_json(train_data, train_output_path)

    
    if val_data:
        val_output_path = os.path.join(args.output_dir, f'{args.task_name}_validation.json')
        save_to_json(val_data, val_output_path)

    
    if test_data:
        test_output_path = os.path.join(args.output_dir, f'{args.task_name}_test.json')
        save_to_json(test_data, test_output_path)

if __name__ == "__main__":
    
    import math
    main()