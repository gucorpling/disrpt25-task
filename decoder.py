import disrptdata
import util
import os
import pandas as pd
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import wandb 


PROMPT = "You will be given two sentences. Please identify the discourse relation between them from the following set of options: contrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction. \nPlease output only the discourse relation label you choose.\n"
MAX_LENGTH = 32768
MAX_RETRIES = 30
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

def preprocess_for_finetuning(example, tokenizer):
    input = f"The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}"
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

def train():
    wandb.init(project="qwen3-finetune", name="qwen3-1.7B", resume="allow")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto", torch_dtype="auto")
    model.enable_input_require_grads()

    dataset = disrptdata.get_combined_dataset()
    
    train_ds = dataset['train']
    train_dataset = train_ds.map(
        partial(preprocess_for_finetuning, tokenizer=tokenizer),
        remove_columns=train_ds.column_names
    )

    eval_ds = dataset['dev']
    eval_dataset = eval_ds.map(
        partial(preprocess_for_finetuning, tokenizer=tokenizer),
        remove_columns=eval_ds.column_names
    )

    args = TrainingArguments(
        output_dir="./output/Qwen3-1.7B",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        num_train_epochs=2,
        save_strategy="epoch",
        # save_steps=300,
        learning_rate=1e-4,
        save_on_each_node=False,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="qwen3-1.7B",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    wandb.finish()

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
            max_new_tokens=MAX_LENGTH,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if response.lower() in LABELS:
            return response

    return "Unknown"

def eval():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto", torch_dtype="auto")
    # model = AutoModelForCausalLM.from_pretrained("output/Qwen3-1.7B/checkpoint-3159", device_map="auto", torch_dtype="auto")

    dataset = disrptdata.get_combined_dataset()
    eval_ds = dataset.get('dev', [])

    preds, labels = [], []

    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for example in tqdm(eval_ds, desc="Evaluating"):
        input_text = f"The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}"
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": input_text}
        ]
        pred = predict(messages, model, tokenizer).strip()
        gold = example['label'].strip()

        example['prediction'] = pred
        preds.append(pred)
        labels.append(gold)

        is_correct = int(pred == gold)

        lang = example.get('lang', 'unknown')
        typ = example.get('framework', 'unknown')
        corpus = example.get('corpus', 'unknown')

        group_stats['all']['correct'] += is_correct
        group_stats['all']['total'] += 1

        group_stats[f'lang={lang}']['correct'] += is_correct
        group_stats[f'lang={lang}']['total'] += 1

        group_stats[f'type={typ}']['correct'] += is_correct
        group_stats[f'type={typ}']['total'] += 1

        group_stats[f'lang={lang}|type={typ}|corpus={corpus}']['correct'] += is_correct
        group_stats[f'lang={lang}|type={typ}|corpus={corpus}']['total'] += 1

    print("Accuracy by group:")
    for group, stats in group_stats.items():
        acc = stats['correct'] / stats['total'] * 100
        print(f"{group}: {acc:.2f}% ({stats['correct']}/{stats['total']})")



if __name__ == "__main__":
    # train()
    eval()




