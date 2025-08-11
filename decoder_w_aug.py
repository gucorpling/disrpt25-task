import argparse
import disrptdata
import util
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
import csv
import random
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
from datasets import Dataset, concatenate_datasets
from functools import partial
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
import wandb
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

PROMPT = "## Role and Goal:\nYou are an expert in discourse analysis, tasked with identifying the discourse relation between two sentence units based on the provided label. Your goal is to accurately determine the relationship between these two units.\n\n## Guidelines:\n1. You will receive Unit1 and Unit2. Unit1 appears before Unit2 in the original text.\n2. You will also be informed about the language of these units.\n3. You will also be informed of the corpus from which the data is drawn, which may help guide your analysis.\n4. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n5. You will be informed whether Unit1 and Unit2 are spoken by the same speaker.\n6. You will also be given the distance between Unit1 and Unit2.\n7. You will be provided with the percentage position of Unit1 and Unit2 in the original document.\n8. You will be given the context in which these two units appear.\n9. The direction of the relationship between these two units will be given.\n10. You will be provided with a set of labels representing possible discourse relations. Choose one label that best fits the relationship between Unit1 and Unit2, and output only the chosen label.\n\n## Labels:\ncontrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction"

MAX_LENGTH = 32768
MAX_RETRIES = 1
LABELS = [
    "contrast",
    "condition",
    "mode",
    "organization",
    "frame",
    "temporal",
    "concession",
    "reformulation",
    "comment",
    "query",
    "attribution",
    "alternation",
    "purpose",
    "explanation",
    "elaboration",
    "causal",
    "conjunction"
]
DIRECTION_MAP = {"1>2": "From Unit1 to Unit2.", "1<2": "From Unit2 to Unit1.", "_": "Unknown."}

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed) 

def preprocess_for_finetuning(example, tokenizer):
    direction = DIRECTION_MAP[example['direction']]
    context = f"{example['context'][0]} {example['context'][1]} {example['context'][2]}"
    input = f"## Language:\n{example['lang']}\n\n## Corpus:\n{example['corpus']}## Framework:\n{example['framework']}\n\n## Same Speaker:\n{example['same_speaker']}\n\n## Distance Between Unit1 and Unit2:\n{example['distance']}\n\n## Percentage Position of Unit1:\n{round(example['u1_position'], 4)}\n\n## Percentage Position of Unit2:\n{round(example['u2_position'], 4)}\n\n## Context:\n{context}\n\n## Direction:\n{direction}\n\n## Unit1:\n{example['u1']}\n\n## Unit2:\n{example['u2']}"

    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['label']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels} 

def train(model_path, checkpoint_path):
    set_all_seeds(42)
    wandb.init(project="qwen3-finetune", name="qwen3-4B", resume="allow", mode="offline")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

    model.enable_input_require_grads()

    augmented_dataset = disrptdata.get_combined_dataset(1, 30, include_common_features=True, include_noncommon_features=True, data_type="aug")
    original_dataset = disrptdata.get_combined_dataset(1, 30, include_common_features=True, include_noncommon_features=True, data_type="orig")
    
    augmented_train = augmented_dataset['train'].shuffle(seed=42)
    original_train = original_dataset['train'].shuffle(seed=42)

    train_ds = concatenate_datasets([augmented_train, original_train])
    train_dataset = train_ds.map(
        partial(preprocess_for_finetuning, tokenizer=tokenizer),
        remove_columns=train_ds.column_names
    )
    
    eval_ds = original_dataset['dev']
    eval_dataset = eval_ds.map(
        partial(preprocess_for_finetuning, tokenizer=tokenizer),
        remove_columns=eval_ds.column_names
    )

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    use_distributed = dist.is_available() and world_size > 1
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=False
    ) if use_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1, 
        sampler=train_sampler,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    args = TrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="epoch",
        learning_rate=5e-5,
        save_on_each_node=False,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="qwen3-4B",
        warmup_steps=200,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.get_train_dataloader = lambda: train_dataloader

    trainer.train()

    wandb.finish()

def save_accuracy_to_csv(group_stats: dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, "accuracy.csv")

    total_accuracy = 0.0
    total_groups = len(group_stats)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group', 'Accuracy (%)', 'Correct', 'Total'])

        for group, stats in group_stats.items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
            writer.writerow([group, f"{acc:.2f}", stats['correct'], stats['total']])

            total_accuracy += acc  

        macro_average = total_accuracy / total_groups if total_groups > 0 else 0.0
        writer.writerow(['Macro-Average', f"{macro_average:.2f}", '', ''])

    print(f"Accuracy report saved to: {csv_filename}")
    print(f"Macro Average Accuracy: {macro_average:.2f}")

def save_confusion_matrices_to_csv(group_stats: dict, output_dir):
    """
    Save confusion matrices for each group in group_stats to individual CSV files.
    """
    output_dir = os.path.join(output_dir, "confusions")
    os.makedirs(output_dir, exist_ok=True)

    # Get global label set
    all_labels = set()
    for stats in group_stats.values():
        all_labels.update(stats['labels'])
        all_labels.update(stats['preds'])
    all_labels = sorted(all_labels)

    for group, stats in group_stats.items():
        y_true = stats['labels']
        y_pred = stats['preds']

        if len(set(y_true + y_pred)) <= 1:
            print(f"Skipping group '{group}' due to insufficient label variation.")
            continue

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        filename = os.path.join(output_dir, f"{group}_confusion.csv")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([""] + all_labels)  # Header row
            for label, row in zip(all_labels, cm):
                writer.writerow([label] + list(row))
        print(f"Confusion report saved.")

def write_predictions_to_rels(eval_ds, output_path, original_rels_path):
    with open(original_rels_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    if len(eval_ds) != (len(original_lines) - 1):
        raise ValueError("The number of predictions does not match the number of data lines in the .rels file")

    new_lines = [original_lines[0].strip()]  

    for example, line in zip(eval_ds, original_lines[1:]): 
        pred_label = example.get('prediction', 'UNKNOWN')
        line = line.strip()

        if not line:
            new_lines.append(line)
            continue

        parts = line.split('\t')
        parts[-1] = pred_label  
        new_line = '\t'.join(parts)
        new_lines.append(new_line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines) + '\n')  

    print(f"Saved predictions to {output_path}")

        
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    for attempt in range(MAX_RETRIES):
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # fix warning
            max_new_tokens=20,
            # temperature=1.0,
            do_sample=False
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if response.lower() in LABELS:
            return response

    return np.random.choice(LABELS)

def eval(checkpoint_path, res_path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", torch_dtype="auto")

    dataset_list = disrptdata.get_list_of_dataset_from_data_dir("data/", data_type="orig")
    for dataset_name in dataset_list:
        dataset = disrptdata.get_dataset(dataset_name, 1, 30, include_common_features=True, include_noncommon_features=True)
        eval_ds = dataset.get('test', [])
        if not eval_ds:
            continue  

        predictions = []

        for example in tqdm(eval_ds, desc="Evaluating"):
            direction = DIRECTION_MAP[example['direction']]
            context = f"{example['context'][0]} {example['context'][1]} {example['context'][2]}"
            input = f"## Language:\n{example['lang']}\n\n## Corpus:\n{example['corpus']}## Framework:\n{example['framework']}\n\n## Same Speaker:\n{example['same_speaker']}\n\n## Distance Between Unit1 and Unit2:\n{example['distance']}\n\n## Percentage Position of Unit1:\n{round(example['u1_position'], 4)}\n\n## Percentage Position of Unit2:\n{round(example['u2_position'], 4)}\n\n## Context:\n{context}\n\n## Direction:\n{direction}\n\n## Unit1:\n{example['u1']}\n\n## Unit2:\n{example['u2']}"

            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": input}
            ]
            pred = predict(messages, model, tokenizer).strip()
            predictions.append(pred)
        
        eval_ds = eval_ds.add_column("prediction", predictions)
        
        output_file = f"{res_path}{dataset_name}/{dataset_name}_test.rels"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_predictions_to_rels(eval_ds, output_file, f"data/{dataset_name}/{dataset_name}_test.rels")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument(
        '--mode', type=str, choices=['train', 'eval'], required=True,
        help="Mode to run: 'train' for training, 'eval' for evaluation"
    )
    parser.add_argument(
        '--model_path', type=str,
        default='JuNymphea/Georgetown-qwen3-4B-pruned-for-disrpt2025',
        help="Path to the model to be used for training"
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default='checkpoint/',
        help="Path where the checkpoint will be saved or loaded from"
    )
    parser.add_argument(
        '--res_path', type=str, default='res/',
        help="Path where the results will be saved (used in eval mode)"
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'train':
        train(args.model_path, args.checkpoint_path)

    elif args.mode == 'eval':
        checkpoint_path = args.checkpoint_path
        eval(checkpoint_path, args.res_path)
