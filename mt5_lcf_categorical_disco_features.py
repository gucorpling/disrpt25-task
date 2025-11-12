import os
import sys
from transformers import MT5Tokenizer, MT5EncoderModel
import torch
import torch.nn as nn
import datasets
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score, f1_score
import argparse
import numpy as np
import csv
from collections import defaultdict
import disrptdata


# Load and combine datasets
combined = disrptdata.get_combined_dataset(include_common_features=True, data_type="all")
from datasets import DatasetDict
combined = DatasetDict({k: v for k, v in combined.items() if v is not None})

print("Available splits:", combined.keys())
print("Train examples:", combined["train"].num_rows)
print("Dev examples:", combined["dev"].num_rows)

train_dataset = combined["train"].class_encode_column("label")
dev_dataset = combined["dev"].class_encode_column("label")
test_dataset = combined["test"].class_encode_column("label")

# === Tokenize dataset ===
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-xl")

# Collect special token candidates
langs = set(combined['train']['lang'])
frameworks = set(combined['train']['framework'])
corpora = set(combined['train']['corpus'])

# Meta tokens
lang_tokens = [f"LANG_{l}" for l in langs]
framework_tokens = [f"FW_{f}" for f in frameworks]
corpus_tokens = [f"CORP_{c}" for c in corpora]

# Genre tokens (if exists)
if "genre" in combined['train'].column_names:
    genres = set(combined['train']['genre'])
    genre_tokens = [f"GENRE_{g}" for g in genres]
else:
    genre_tokens = []

# Categorical feature tokens
disco_tokens = [
    "IS_SENTENCE_0", "IS_SENTENCE_1",
    "DISCONTINUOUS_0", "DISCONTINUOUS_1",
    "SAME_SPEAKER_0", "SAME_SPEAKER_1"
]

special_tokens = lang_tokens + framework_tokens + corpus_tokens + genre_tokens + disco_tokens + ["[SEP]"]
tokenizer.add_tokens(special_tokens)

def preprocess_input(batch):
    texts = []
    for i, (lang, fw, corpus, u1, u2, direction) in enumerate(zip(
        batch["lang"], batch["framework"], batch["corpus"],
        batch["u1"], batch["u2"], batch["direction"]
    )):
        meta = f"LANG_{lang} FW_{fw} CORP_{corpus}"

        features = []
        if "u1_issent" in batch:
            features.append(f"IS_SENTENCE_{int(batch['u1_issent'][i])}")
        if "u1_discontinuous" in batch:
            features.append(f"DISCONTINUOUS_{int(batch['u1_discontinuous'][i])}")
        if "same_speaker" in batch:
            features.append(f"SAME_SPEAKER_{int(batch['same_speaker'][i])}")
        if "genre" in batch and batch["genre"][i] is not None:
            features.append(f"GENRE_{batch['genre'][i]}")

        disco = " ".join(features)

        if direction == '1>2':
            span = f"}} {u1} > Arg2: {u2}"
        elif direction == '1<2':
            span = f"{u1} < Arg2: {u2} {{"
        else:
            span = f"{u1} Arg2: {u2}"

        text = f"{meta} [SEP] {disco} [SEP] {span}"
        texts.append(text)

    return {"text": texts, "label": batch["label"]}

# === Tokenize ===
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_text = train_dataset.map(preprocess_input, batched=True)
dev_text = dev_dataset.map(preprocess_input, batched=True)
test_text = test_dataset.map(preprocess_input, batched=True)

print("Sample input text:\n", train_text[0]["text"])

train_tokenized = train_text.map(tokenize, batch_size=8, batched=True)
dev_tokenized  = dev_text.map(tokenize, batch_size=8, batched=True)
test_tokenized  = test_text.map(tokenize, batch_size=8, batched=True)

train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dev_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# === Model Definition ===
class MT5Classifier(nn.Module):
    def __init__(self, num_labels, tokenizer_len):
        super().__init__()
        self.encoder = MT5EncoderModel.from_pretrained("google/mt5-xl")
        self.encoder.resize_token_embeddings(tokenizer_len)
        hidden_size = self.encoder.config.d_model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

num_labels = train_tokenized.features["label"].num_classes
model = MT5Classifier(num_labels=num_labels, tokenizer_len=len(tokenizer))

# === Training Setup ===
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

training_args = TrainingArguments(
    output_dir="output/MT5_1",
    overwrite_output_dir=False,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    save_safetensors=False,
    # Enable distributed training (multi-GPU)
    deepspeed=None,  # Optional: Use DeepSpeed for larger models, if needed
    fp16=True,  # Mixed precision training (recommended for multi-GPU)
    local_rank=-1,  # Used by torchrun for multi-GPU
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    # Get label names from the feature
    label_names = train_tokenized.features["label"].names

    report = classification_report(labels, preds, target_names=label_names)
    print("\n=== Classification Report ===")
    print(report)
    return {"accuracy": acc, "f1": f1}

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save misclassified examples
pred_out = trainer.predict(test_tokenized)
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
out_path = "intermediate/MT5_1.csv"  # Replace with your desired path

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "true_label", "pred_label", "confidence"])
    writer.writerows(mis)

print(f"✅ Saved {len(mis)} misclassified examples to:\n{out_path}")

# Accuracy per dataset
group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for i, ex in enumerate(test_dataset):
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
