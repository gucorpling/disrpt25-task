# this script is used for encoder-only MT5 model with a language embedding layer and a classifier head.

from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import torch.nn as nn
# from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from run_model import load_rels_dataset
import argparse
import numpy as np


# === load dataset ===
dataset = load_rels_dataset(
    "data/eng.erst.gum/eng.erst.gum_dev.rels",
    "data/eng.erst.gum/eng.erst.gum_train.rels"
)

# we only use the dev set for training and testing in prototype
dev_only = dataset["dev"]
split_dataset = dev_only.train_test_split(test_size=0.2, seed=42)

# encode labels to integers
label_encoder = LabelEncoder()
label_encoder.fit(split_dataset["train"]["label"])

def encode_label(example):
    example["label"] = label_encoder.transform([example["label"]])[0]
    return example

split_dataset = split_dataset.map(encode_label)
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

def preprocess(example):
    # combine the two spans into a single text input
    text = f"Classify: Arg1: {example['u1']} Arg2: {example['u2']}"
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    encoded["label"] = example["label"]
    return encoded

# drop the original columns that are not needed for training
tokenized_dataset = split_dataset.map(
    preprocess,
    remove_columns=split_dataset["train"].column_names
)
tokenized_dataset.set_format("torch")

# === model setup ===
class MT5Classifier(nn.Module):
    def __init__(self, num_labels, num_languages=1, lang_emb_dim=None):
        super().__init__()
        self.encoder = MT5ForConditionalGeneration.from_pretrained("google/mt5-base", use_safetensors=True).get_encoder()
        hidden_size = self.encoder.config.d_model
        self.lang_emb_dim = lang_emb_dim or hidden_size
        self.language_embedding = nn.Embedding(num_languages, self.lang_emb_dim)
        self.lang_proj = nn.Linear(self.lang_emb_dim, hidden_size) if self.lang_emb_dim != hidden_size else None
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, language_ids=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids)
            if self.lang_proj:
                lang_emb = self.lang_proj(lang_emb)
            pooled = pooled + lang_emb
        logits = self.classifier(pooled)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

model = MT5Classifier(num_labels=len(label_encoder.classes_))

# === training setup ===
parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", type=str, default="yes")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda == "yes" else "cpu")
model = model.to(device)

training_args = TrainingArguments(
    output_dir="baseline_mt5_classifier.results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, target_names=label_encoder.classes_)

    print("\n=== Classification Report ===")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}\n")

    return {
        "accuracy": acc,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# === error analysis ===
# get predictions on dev set
pred_out = trainer.predict(tokenized_dataset["dev"])
logits = pred_out.predictions
labels = pred_out.label_ids

# convert logits to label IDs
preds = np.argmax(logits, axis=1)

# softmax confidence score 
probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

# extract missclassified examples 
dev_texts = split_dataset["dev"]
formatted_texts = [
    f"Arg1: {u1} | Arg2: {u2}"
    for u1, u2 in zip(dev_texts["u1"], dev_texts["u2"])
]

mis = [
    (text, label_encoder.classes_[true], label_encoder.classes_[pred], probs[i][pred])
    for i, (text, true, pred) in enumerate(zip(formatted_texts, labels, preds))
    if true != pred
]

# log the results to a csv file
with open("misclassifications.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "true_label", "pred_label", "confidence"])
    writer.writerows(mis)

print(f"✅ Logged {len(mis)} misclassified examples to misclassifications.csv")