import json
import argparse
import random
import os
from glob import glob

def merge_json_files(input_patterns, output_file):
    """
    Merges multiple JSON files (matching glob patterns) into a single file.

    Args:
        input_patterns (list of str): A list of file paths or glob patterns 
                                      (e.g., ['data/bace/train.json', 'data/hiv/train.json']).
        output_file (str): The path to the output merged JSON file.
    """
    merged_data = []
    
    
    for pattern in input_patterns:
        
        file_paths = glob(pattern)
        if not file_paths:
            print(f"⚠Warning: No files found for pattern '{pattern}'. Skipping.")
            continue
            
        for file_path in file_paths:
            print(f"Reading from {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        print(f"Warning: File {file_path} does not contain a JSON list. Skipping.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
            except Exception as e:
                print(f"An error occurred with file {file_path}: {e}")

    
    random.shuffle(merged_data)
    
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    
    print(f"\nSaving {len(merged_data)} merged records to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
        
    print("Merging completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple JSON dataset files into one.")
    
    parser.add_argument(
        '--inputs', 
        nargs='+', 
        required=True, 
        help="A list of input JSON files or glob patterns to merge. \
              Example: --inputs data/bace/train.json data/hiv/train.json"
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help="Path to the output merged JSON file. \
              Example: --output data/multitask/train.json"
    )
    
    args = parser.parse_args()
    
    merge_json_files(args.inputs, args.output)

if __name__ == "__main__":
    main()