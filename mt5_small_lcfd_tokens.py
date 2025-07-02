# -*- coding: utf-8 -*-
"""mt5_small_LCFD_tokens.py

Converted from Colab notebook

"""

import os
import torch
import torch.nn as nn
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import datasets
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import defaultdict
from safetensors.torch import load_file
import numpy as np
import csv
import disrptdata



# === load dataset ===
"""
experiment with two languages: Chinese and English
"""

# load and combine datasets
# zho = disrptdata.get_dataset("zho.rst.gcdt")
# eng = disrptdata.get_dataset("eng.erst.gum")
combined = disrptdata.get_combined_dataset()
print("Train examples:", combined["train"].num_rows)
print("Dev examples:", combined["dev"].num_rows)

train_dataset = combined['train']
dev_dataset = combined['dev']
train_dataset = train_dataset.class_encode_column('label')
dev_dataset   = dev_dataset.class_encode_column('label')

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")

# add features as tokens
langs = set(combined['train']['lang'])
frameworks = set(combined['train']['framework'])
corpora = set(combined['train']['corpus'])

lang_tokens = [f"[LANG:{l}]" for l in langs]
framework_tokens = [f"[FW:{f}]" for f in frameworks]
corpus_tokens = [f"[CORP:{c}]" for c in corpora]

meta_tokens = lang_tokens + framework_tokens + corpus_tokens
tokenizer.add_tokens(meta_tokens)

def preprocess(batch):
    # encode LCF features (language, corpus, framework)
    texts = []
    for lang, fw, corpus, u1, u2, direction in zip(
        batch["lang"], batch["framework"], batch["corpus"],
        batch["u1"], batch["u2"], batch["direction"]
    ):
        meta = f"[LANG:{lang}] [FW:{fw}] [CORP:{corpus}]"

        if direction == '1>2':
            span = f"Arg1: }} {u1} > [SEP] Arg2: {u2}" # }}
        elif direction == '1<2':
            span = f"Arg1: {u1} < Arg2: {u2} {{"
        else:
            span = f"Arg1: {u1} [SEP] Arg2: {u2}"

        text = f"Classify: {meta} {span}"
        texts.append(text)

    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=512)
    encoded["label"] = batch["label"]
    return encoded

train_tokenized = train_dataset.map(preprocess, batched=True)
dev_tokenized   = dev_dataset.map(preprocess, batched=True)

train_tokenized.set_format('torch', columns=["input_ids", "attention_mask", "label"])
dev_tokenized.set_format('torch', columns=["input_ids", "attention_mask", "label"])

class MT5Classifier(nn.Module):
    def __init__(self, num_labels, tokenizer_len):
        super().__init__()
        base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
        base_model.resize_token_embeddings(tokenizer_len)
        self.encoder = base_model.get_encoder()
        # classifier head
        hidden_size = base_model.config.d_model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

num_labels = train_tokenized.features["label"].num_classes
model = MT5Classifier(num_labels=num_labels, tokenizer_len=len(tokenizer))

# # load checkpoint 
# checkpoint_path = "./output/mt5large/checkpoint-7316/model.safetensors"
# state_dict = load_file(checkpoint_path)
# model.load_state_dict(state_dict, strict=False)

# === training setup ===
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

training_args = TrainingArguments(
    output_dir="./output/mt5large-without-augment",  # Replace with your desired output path
    overwrite_output_dir=False,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    auto_find_batch_size=True,
)

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    # get label names from the feature
    label_names = train_tokenized.features["label"].names

    report = classification_report(labels, preds, target_names=label_names)
    print("\n=== Classification Report ===")
    print(report)
    return {"accuracy": acc, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

batch = next(iter(trainer.get_train_dataloader()))
print("Batch input_ids shape:", batch["input_ids"].shape)

# trainer.train()
# trainer.evaluate()

# === error analysis ===
# Get predictions on tokenized dev set
pred_out = trainer.predict(dev_tokenized)
logits = pred_out.predictions
labels = pred_out.label_ids
preds = np.argmax(logits, axis=1)
probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

# Use label names from the dataset
label_names = dev_dataset.features["label"].names

# Use original dev_dataset (not tokenized) for u1 and u2
formatted_texts = [
    f"Arg1: {ex['u1']} | Arg2: {ex['u2']}" for ex in dev_dataset
]

# Extract misclassified examples
mis = [
    (text, label_names[true], label_names[pred], probs[i][pred])
    for i, (text, true, pred) in enumerate(zip(formatted_texts, labels, preds))
    if true != pred
]

# Save to disk
out_path = "misclassifications/misclassifications_LCFD.csv"  # Replace with your desired path

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "true_label", "pred_label", "confidence"])
    writer.writerows(mis)

print(f"✅ Saved {len(mis)} misclassified examples to:\n{out_path}")

# accuracy per dataset
group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for i, ex in enumerate(dev_dataset):
    key = (ex["lang"], ex["framework"], ex["corpus"])
    true = labels[i]
    pred = preds[i]
    group_stats[key]["total"] += 1
    if true == pred:
        group_stats[key]["correct"] += 1

print("\n=== Accuracy per (lang, framework, corpus) group ===")
for key, stats in sorted(group_stats.items()):
    lang, fw, corpus = key
    correct = stats["correct"]
    total = stats["total"]
    acc = correct / total if total > 0 else 0.0
    print(f"[{lang}, {fw}, {corpus}]: Accuracy = {acc:.4f} ({correct}/{total})")