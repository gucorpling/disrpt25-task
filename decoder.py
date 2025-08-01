import disrptdata
import util
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
import csv
import random
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from datasets import Dataset
from functools import partial
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
import wandb 


# PROMPT = "You will be given two sentences. Please identify the discourse relation between them from the following set of options: contrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction. \nPlease output only the discourse relation label you choose.\n"
# PROMPT = "You are an expert in discourse analysis. Given two sentences and their directional relationship, please identify the discourse relation between them from the following set of options: contrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction. Please output only the label of the discourse relation you choose."
# PROMPT = "## Role and Goal:\nYou are an expert in discourse analysis, tasked with identifying the discourse relation between two sentence spans based on the provided label. Your goal is to accurately determine the relationship between these two sentences.\n\n## Guidelines:\n1. You will receive Sentence 1 and Sentence 2. Sentence 1 appears before Sentence 2 in the original text.\n2. You will also be informed about the language of these sentences.\n3. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n4. The direction of the relationship between these two sentences will be given.\n5. You will be provided with a set of labels representing possible discourse relations. Choose one label that best fits the relationship between Sentence 1 and Sentence 2, and output only the chosen label.\n\n## Labels:\ncontrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction"
# PROMPT = "## Role and Goal:\nYou are an expert in discourse analysis, tasked with identifying the discourse relation between two sentence spans based on the provided label. Your goal is to accurately determine the relationship between these two sentences.\n\n## Guidelines:\n1. You will receive Sentence 1 and Sentence 2. Sentence 1 appears before Sentence 2 in the original text.\n2. You will also be informed about the language of these sentences.\n3. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n4. The direction of the relationship between these two sentences will be given.\n5. You will be given the context in which these two sentences appear.\n6. You will be provided with a set of labels representing possible discourse relations. Choose one label that best fits the relationship between Sentence 1 and Sentence 2, and output only the chosen label.\n\n## Labels:\ncontrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction"
# PROMPT = "## Role and Goal:\nYou are an expert in discourse analysis, tasked with identifying the discourse relation between two sentence spans based on the provided label. Your goal is to accurately determine the relationship between these two sentences.\n\n## Guidelines:\n1. You will receive Sentence 1 and Sentence 2. Sentence 1 appears before Sentence 2 in the original text.\n2. You will also be informed about the language of these sentences.\n3. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n4. The direction of the relationship between these two sentences will be given.\n5. You will be given the context in which these two sentences appear.\n6. You will be provided with a set of labels representing possible discourse relations. Choose one label that best fits the relationship between Sentence 1 and Sentence 2, and output only the chosen label.\n\n## Labels:\ncontrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction\n\n---\n\n### Example 1:\n\n**Input:** \n\n## Language:\neng\n\n## Framework:\nRST\n\n## Direction:\nFrom Sentence2 to Sentence1.\n\n## Context:\nOur experience of art develops from the interaction of several cognitive and affective processes; the beginning of which is a visual scan of the artwork . When regarding an artwork, a viewer gathers information through a series fixations, interspersed by rapid movements of the eye called saccades. The direction of saccades is determined by an interaction between the goals of the observer and the physical properties of the different elements of the scene (e.g. colour, texture, brightness etc). Importantly, studying eye movements offers an insight that does not depend on the participants’ beliefs, memories or subjective impressions of the artwork. Previous eye tracking research has highlighted the potential to transform the ways we understand visual processing in the arts (see for example Brieber 2014; Binderman et al., 2005) and at the same time offers a direct way of studying several important factors of a museum visit (Filippini Fantoni et al., 2013; Heidenreich & Turano 2011; Milekic 2010).\n\n## Sentence1:\nThe direction of saccades is determined by an interaction between the goals of the observer and the physical properties of the different elements of the scene\n\n## Sentence2:\nImportantly, studying eye movements offers an insight\n\n**Output:**\ncomment\n\n---\n\n### Example 2:\n\n**Input:** \n\n## Language:\nzho\n\n## Framework:\ndep\n\n## Direction:\nFrom Sentence1 to Sentence2.\n\n## Context:\n随着认知计算的飞速发展,通用知识图谱的自动构建取得了极大的进步,但在垂直领域由于缺乏本体等语义信息,导致进展缓慢。叙词表广泛分布于各个专业领域且蕴藏着丰富的语义信息,如能对这些语义信息进行合理的提取和利用,必然能在一定程度上帮助领域知识图谱的自动构建。\n\n## Sentence1:\n随着认知计算的飞速发展,\n\n## Sentence2:\n通用知识图谱的自动构建取得了极大的进步,\n\n**Output:**\nframe\n\n---\n\n### Example 3:\n\n**Input:** \n\n## Language:\ndeu\n\n## Framework:\npdtb\n\n## Direction:\nFrom Sentence2 to Sentence1.\n\n## Context:\nWas nützt ihnen die Verlagerung an einen neuen Standort mit den längeren Öffnungszeiten , wenn sie schlecht zu Fuß sind ? Da bleibt nur der Hinweis , sich den Zustellerinnen zuzuwenden , die selber schon so manche Dienstleistung übernehmen . Fragen kostet ja nichts .\n\n## Sentence1:\nDa bleibt nur der Hinweis , sich den Zustellerinnen zuzuwenden , die selber schon so manche Dienstleistung übernehmen .\n\n## Sentence2:\nFragen kostet ja nichts . \n\n**Output:**\ncausal\n\n---"
PROMPT = "## Role and Goal:\nYou are an expert in discourse analysis, tasked with identifying the discourse relation between two sentence units based on the provided label. Your goal is to accurately determine the relationship between these two units.\n\n## Guidelines:\n1. You will receive Unit1 and Unit2. Unit1 appears before Unit2 in the original text.\n2. You will also be informed about the language of these units.\n3. You will also be informed of the corpus from which the data is drawn, which may help guide your analysis.\n4. The framework for analysis will be provided, outlining the structure used for discourse analysis.\n5. You will be given the context in which these two units appear.\n6. The direction of the relationship between these two units will be given.\n7. You will be provided with a set of labels representing possible discourse relations. Choose one label that best fits the relationship between Unit1 and Unit2, and output only the chosen label.\n\n## Labels:\ncontrast, condition, mode, organization, frame, temporal, concession, reformulation, comment, query, attribution, alternation, purpose, explanation, elaboration, causal, conjunction"

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
# DIRECTION_MAP = {"1>2": "from Sentence1 to Sentence 2", "1<2": "from Sentence2 to Sentence 1", "_": "unknown"}
# DIRECTION_MAP = {"1>2": "From Sentence1 to Sentence 2.", "1<2": "From Sentence2 to Sentence 1.", "_": "Unknown."}
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
    # input = f"The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}"
    # input = f"## Sentence1:{example['u1']}\n## Sentence2:{example['u2']}\n## Direction:{example['direction']}"

    direction = DIRECTION_MAP[example['direction']]
    # input = f"The direction is {direction}. The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}."
    # input = f"## Language:\n{example['lang']}\n\n## Framework:\n{example['framework']}\n\n## Direction:\n{direction}\n\n## Sentence1:\n{example['u1']}\n\n## Sentence2:\n{example['u2']}"
    context = f"{example['context'][0]} {example['context'][1]} {example['context'][2]}"
    # input = f"## Language:\n{example['lang']}\n\n## Framework:\n{example['framework']}\n\n## Direction:\n{direction}\n\n## Context:\n{context}\n\n## Sentence1:\n{example['u1']}\n\n## Sentence2:\n{example['u2']}"
    input = f"## Language:\n{example['lang']}\n\n## Corpus:\n{example['corpus']}## Framework:\n{example['framework']}\n\n## Context:\n{context}\n\n## Direction:\n{direction}\n\n## Unit1:\n{example['u1']}\n\n## Unit2:\n{example['u2']}"

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
    set_all_seeds(42)
    wandb.init(project="qwen3-finetune", name="qwen3-4B", resume="allow")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained(f"output/Qwen3-4B-pruned-36to35-1prune_layers", torch_dtype="auto")

    model.enable_input_require_grads()

    dataset = disrptdata.get_combined_dataset(0, 30, include_common_features=False, include_noncommon_features=False, data_type="orig")
    
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
        output_dir="./output/Qwen3-4B-markdown-context-hyper-corpus",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="epoch",
        # eval_steps=50,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="epoch",
        # save_steps=300,
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
            # temperature=1.0,
            do_sample=False,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if response.lower() in LABELS:
            return response

    return "Unknown"

def eval():
    checkpoint_name = "Qwen3-4B-markdown-context-hyper-corpus/checkpoint-3666"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=False, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto", torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained(f"output/{checkpoint_name}", device_map="auto", torch_dtype="auto")

    dataset = disrptdata.get_combined_dataset(0, 30, include_common_features=False, include_noncommon_features=False, data_type="orig")
    eval_ds = dataset.get('test', [])

    preds, labels = [], []

    group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for example in tqdm(eval_ds, desc="Evaluating"):
        # input = f"The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}"
        direction = DIRECTION_MAP[example['direction']]
        # input = f"The sentences are:\n\nSentence 1: {example['u1']}\n\nSentence 2: {example['u2']}. The direction is {direction}."
        # input = f"## Language:\n{example['lang']}\n\n## Framework:\n{example['framework']}\n\n## Direction:\n{direction}\n\n## Sentence1:\n{example['u1']}\n\n## Sentence2:\n{example['u2']}"
        context = f"{example['context'][0]} {example['context'][1]} {example['context'][2]}"
        # input = f"## Language:\n{example['lang']}\n\n## Framework:\n{example['framework']}\n\n## Direction:\n{direction}\n\n## Context:\n{context}\n\n## Sentence1:\n{example['u1']}\n\n## Sentence2:\n{example['u2']}"
        input = f"## Language:\n{example['lang']}\n\n## Corpus:\n{example['corpus']}## Framework:\n{example['framework']}\n\n## Context:\n{context}\n\n## Direction:\n{direction}\n\n## Unit1:\n{example['u1']}\n\n## Unit2:\n{example['u2']}"
        
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": input}
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

        # group_stats[f'lang={lang}']['correct'] += is_correct
        # group_stats[f'lang={lang}']['total'] += 1

        # group_stats[f'type={typ}']['correct'] += is_correct
        # group_stats[f'type={typ}']['total'] += 1

        group_stats[f'lang={lang}|type={typ}|corpus={corpus}']['correct'] += is_correct
        group_stats[f'lang={lang}|type={typ}|corpus={corpus}']['total'] += 1

    csv_filename = f"intermediate/{checkpoint_name.split('/')[0]}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group', 'Accuracy (%)', 'Correct', 'Total'])  # Write header
        
        for group, stats in group_stats.items():
            acc = stats['correct'] / stats['total'] * 100
            writer.writerow([group, f"{acc:.2f}", stats['correct'], stats['total']])

    print(f"Results have been saved to {csv_filename}")

    acc = group_stats['all']['correct'] / group_stats['all']['total'] * 100
    print(f"All:{acc}")
    
if __name__ == "__main__":
    train()
    # eval()
