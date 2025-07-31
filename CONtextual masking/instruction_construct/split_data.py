import argparse
import json
import random
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_json_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a JSON dataset into training, validation, and test sets.
    """
    # 检查比例总和是否为1
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    print(f"Reading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 仅在数据足够时才进行随机打乱，以保持可复现性
    if len(data) > 1:
        random.shuffle(data)

    # 第一次划分：训练集 vs. (验证集 + 测试集)
    try:
        train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42)
    except ValueError:
         # 如果数据量太小无法划分，则将所有数据放入训练集
        print("Warning: Dataset too small to split. Placing all data in the training set.")
        return data, [], []

    # 如果剩余数据量太小，无法再分
    if len(temp_data) < 2:
        return train_data, temp_data, []

    # 第二次划分：验证集 vs. 测试集
    # 计算验证集在剩余数据中的比例
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
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} samples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Split a JSON dataset into named train, val, and test files.")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split files.")
    # --- 新增参数 ---
    parser.add_argument('--task_name', type=str, required=True, help="Name of the task to prefix output files (e.g., 'bace', 'hiv').")
    
    args = parser.parse_args()

    # 进行数据集划分
    train_data, val_data, test_data = split_dataset(args.input_json)
    
    print(f"\nSplitting complete for task '{args.task_name}':")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # --- 关键修改：使用task_name构建输出文件名 ---
    # 保存训练集文件
    train_output_path = os.path.join(args.output_dir, f'{args.task_name}_train.json')
    save_to_json(train_data, train_output_path)

    # 保存验证集文件 (如果存在)
    if val_data:
        val_output_path = os.path.join(args.output_dir, f'{args.task_name}_validation.json')
        save_to_json(val_data, val_output_path)

    # 保存测试集文件 (如果存在)
    if test_data:
        test_output_path = os.path.join(args.output_dir, f'{args.task_name}_test.json')
        save_to_json(test_data, test_output_path)

if __name__ == "__main__":
    # 增加math库的导入，因为isclose需要它
    import math
    main()