import json
from datasets import Dataset
from typing import List

IGNORE_INDEX = -100  # 不参与 loss 的 token 填充值


class DataCollatorForSupervisedDataset:
    """
    用于处理 batch 中的 padding，确保 input_ids 和 labels 对齐，并且 labels 中非监督区域为 IGNORE_INDEX。
    """
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, batch):
        input_ids = [example['input_ids'] for example in batch]
        labels = [example['labels'] for example in batch]

        batch_input = self.tokenizer.pad({"input_ids": input_ids}, padding=self.padding, return_tensors="pt")
        batch_labels = self.tokenizer.pad({"input_ids": labels}, padding=self.padding, return_tensors="pt")
        batch_input["labels"] = batch_labels["input_ids"]
        return batch_input


def build_instruction_dataset(
    data_path: List[str],
    tokenizer,
    max_seq_length=1024,
    data_cache_dir=None,
    preprocessing_num_workers=4
):
    """
    将 JSON 文件读取为 instruction + input + output 格式，并编码为模型训练所需格式。
    """
    all_data = []
    for path in data_path:
        with open(path, "r") as f:
            samples = json.load(f)
            all_data.extend(samples)

    def preprocess(example):
        # 拼接为 prompt：instruction + input
        prompt = example["instruction"].strip() + "\n" + example["input"].strip()
        target = example["output"].strip()

        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length).input_ids
        target_ids = tokenizer(target, truncation=True, max_length=max_seq_length - len(prompt_ids) - 1).input_ids

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        return {
            "input_ids": input_ids[:max_seq_length],
            "labels": labels[:max_seq_length]
        }

    dataset = Dataset.from_list(all_data)
    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return tokenized_dataset
