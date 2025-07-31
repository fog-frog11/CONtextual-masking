import os
import sys
import torch
import logging
from dataclasses import dataclass, field
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    HfArgumentParser, TrainerCallback,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Logger and Callback (保持不变) ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class CustomGenerationCallback(TrainerCallback):
    # ... (这部分代码保持不变)
    def __init__(self, tokenizer, sample, generation_max_new_tokens=100):
        self.tokenizer = tokenizer
        self.sample = sample
        # 修正Prompt格式以匹配数据预处理
        self.prompt = f"Instruction:\n{sample['instruction']}\n\nInput:\n{sample['input']}\n\nOutput:\n"
        self.true_output = sample['output']
        self.max_new_tokens = generation_max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        logger.info(f"\n--- Custom Generation Sample at Step {state.global_step} ---")
        model.eval()
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        gen = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info("Prompt:\n%s", self.prompt)
        logger.info("Expected Output:\n%s", self.true_output)
        logger.info("Generated Output:\n%s", gen.strip())
        model.train()
        return control

# --- Argument Classes (保持不变) ---
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier"})
    train_file: str = field(metadata={"help": "The input training data file (a json file)."})
    eval_file: str = field(metadata={"help": "The input evaluation data file (a json file)."})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Maximum sequence length."})

# --- 参数解析 ---
parser = HfArgumentParser((ScriptArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    script_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    script_args, training_args = parser.parse_args_into_dataclasses()

# --- !!! 关键修改：强制设置保存策略 !!! ---

# 1. 强制关闭中间检查点保存
training_args.save_strategy = "no"
logger.info("Checkpoint saving during training is DISABLED. The final model will be saved at the end.")

# 2. 评估策略仍然可以保留，以便观察loss变化
if not hasattr(training_args, 'evaluation_strategy') or training_args.evaluation_strategy == "no":
    training_args.evaluation_strategy = "steps"
if not hasattr(training_args, 'eval_steps') or training_args.eval_steps is None:
    # 评估频率可以设高一些，比如每1000步或更少，取决于你的总步数
    training_args.eval_steps = 1000 
    logger.info(f"Evaluation strategy set to 'steps' with eval_steps={training_args.eval_steps}")

# 3. 由于不保存最好模型，以下参数不再需要，但设置也无妨
training_args.load_best_model_at_end = False # 设为False，因为没有中间模型可以加载

# 4. 其他设置保持不变
if training_args.report_to == ["all"] or training_args.report_to is None:
    training_args.report_to = []

if not training_args.fp16 and not training_args.bf16:
    training_args.fp16 = True

# --- Model & Tokenizer (保持不变) ---
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Data Loading & Preprocessing (保持不变) ---
train_dataset = load_dataset("json", data_files={"train": script_args.train_file})["train"]
eval_dataset = load_dataset("json", data_files={"eval": script_args.eval_file})["eval"]

def preprocess_function(examples):
    prompt_template = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nOutput:\n"
    full_texts, prompts = [], []
    for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
        prompt = prompt_template.format(instruction=instruction, input=inp)
        prompts.append(prompt)
        full_texts.append(prompt + output + tokenizer.eos_token)
    
    model_inputs = tokenizer(full_texts, max_length=script_args.max_seq_length, padding="max_length", truncation=True)
    prompt_tokens = tokenizer(prompts, max_length=script_args.max_seq_length, padding="max_length", truncation=True)
    
    labels = []
    for i in range(len(model_inputs['input_ids'])):
        label_ids = list(model_inputs['input_ids'][i])
        try:
            # 找到prompt的实际token长度（不包括padding）
            prompt_len = prompt_tokens['attention_mask'][i].index(0)
        except ValueError:
            prompt_len = len(prompt_tokens['attention_mask'][i])
        
        for j in range(prompt_len):
            label_ids[j] = -100
        labels.append(label_ids)
        
    model_inputs["labels"] = labels
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names, num_proc=os.cpu_count() or 1)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names, num_proc=os.cpu_count() or 1)

# --- Trainer Initialization ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Callbacks
if eval_dataset and len(eval_dataset) > 0:
    raw_eval_sample = load_dataset("json", data_files={"eval": script_args.eval_file})["eval"][0]
    trainer.add_callback(CustomGenerationCallback(tokenizer=tokenizer, sample=raw_eval_sample))

# --- Train and Save ---
logger.info("Starting training...")
trainer.train()

logger.info(f"Saving final model to {training_args.output_dir}")
trainer.save_model(training_args.output_dir)