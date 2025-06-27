import disrptdata
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import wandb

wandb.init(project="qwen3-embedding-classifier", name="qwen3-0.6B-cached", resume="allow")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

embedding_dim = embed_model.get_sentence_embedding_dimension()
input_dim = 4 * embedding_dim
hidden_dim = 256

dataset = disrptdata.get_combined_dataset()

label_list = sorted(set(example["label"] for example in dataset["train"]))
label2id = {label: idx for idx, label in enumerate(label_list)}

lang_list = sorted(set(example["lang"] for example in dataset["train"]))
type_list = sorted(set(example["framework"] for example in dataset["train"]))
domain_list = sorted(set(example["direction"] for example in dataset["train"]))
corpus_list = sorted(set(
    example["corpus"] for split in ["train", "dev"]
    for example in dataset[split]
)) # only for eval

lang2id = {lang: idx for idx, lang in enumerate(lang_list)}
type2id = {typ: idx for idx, typ in enumerate(type_list)}
domain2id = {d: idx for idx, d in enumerate(domain_list)}
corpus2id = {c: idx for idx, c in enumerate(corpus_list)} # only for eval

num_classes = len(label2id)

def load_or_generate_tensor(path, desc, gen_fn):
    if os.path.exists(path):
        print(f"Loading cached {desc} from {path}")
        return torch.load(path)
    else:
        print(f"Generating {desc}...")
        tensor = gen_fn()
        torch.save(tensor, path)
        return tensor

def prepare_embeddings_and_features(hf_dataset, label2id, lang2id, type2id, domain2id, corpus2id, split_name, embed_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fx = f"embeddings/embeddings_{split_name}_x.pt"
    fy = f"embeddings/embeddings_{split_name}_y.pt"
    flang = f"embeddings/embeddings_{split_name}_lang.pt"
    ftype = f"embeddings/embeddings_{split_name}_type.pt"
    fdom = f"embeddings/embeddings_{split_name}_domain.pt"
    fcor = f"embeddings/embeddings_{split_name}_cor.pt"

    x = load_or_generate_tensor(
        fx, "embeddings X",
        lambda: torch.stack([
            torch.cat([
                e1 := embed_model.encode(ex["u1"], convert_to_tensor=True, device=device),
                e2 := embed_model.encode(ex["u2"], convert_to_tensor=True, device=device),
                torch.abs(e1 - e2), e1 * e2
            ], dim=-1).cpu()
            for ex in hf_dataset
        ])
    )

    y = load_or_generate_tensor(
        fy, "labels Y",
        lambda: torch.tensor([label2id[ex["label"]] for ex in hf_dataset])
    )

    lang = load_or_generate_tensor(
        flang, "language features",
        lambda: torch.tensor([lang2id[ex["lang"]] for ex in hf_dataset])
    )

    typ = load_or_generate_tensor(
        ftype, "type features",
        lambda: torch.tensor([type2id[ex["framework"]] for ex in hf_dataset])
    )

    dom = load_or_generate_tensor(
        fdom, "domain features",
        lambda: torch.tensor([domain2id[ex["direction"]] for ex in hf_dataset])
    )

    cor = load_or_generate_tensor(
        fcor, "corpus features",
        lambda: torch.tensor([corpus2id[ex["corpus"]] for ex in hf_dataset])
    )

    return x, y, lang, typ, dom, cor

train_x, train_y, train_langs, train_framework, train_direction, train_corpus=prepare_embeddings_and_features(
    dataset["train"], label2id, lang2id, type2id, domain2id, corpus2id, "train", embed_model)
val_x, val_y, val_langs, val_framework, val_direction, val_corpus=prepare_embeddings_and_features(
    dataset["dev"], label2id, lang2id, type2id, domain2id, corpus2id, "dev", embed_model)

train_dataset = TensorDataset(train_x, train_y, train_langs, train_framework, train_direction, train_corpus)
val_dataset = TensorDataset(val_x, val_y, val_langs, val_framework, val_direction, val_corpus)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, lang_size, type_size, domain_size, corpus_size):
        super().__init__()
        self.lang_embedding = nn.Embedding(lang_size, 16)
        self.type_embedding = nn.Embedding(type_size, 16)
        self.domain_embedding = nn.Embedding(domain_size, 16)
        self.fc1 = nn.Linear(input_dim + 48, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lang, typ, dom):
        lang_embed = self.lang_embedding(lang)
        type_embed = self.type_embedding(typ)
        dom_embed = self.domain_embedding(dom)
        enhanced = torch.cat([x, lang_embed, type_embed, dom_embed], dim=-1)
        return self.fc2(self.dropout(self.relu(self.fc1(enhanced))))

model = Classifier(input_dim, hidden_dim, num_classes,
                   lang_size=len(lang2id), type_size=len(type2id), domain_size=len(domain2id), corpus_size=len(corpus2id)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_x, batch_y, lang, typ, dom, cor in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        lang, typ, dom = lang.to(device), typ.to(device), dom.to(device)
        logits = model(batch_x, lang, typ, dom)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    wandb.log({"epoch": epoch+1, "train_loss": total_loss})

    model.eval()
    preds, labels, langs, types, corpus= [], [], [], [], []

    with torch.no_grad():
        for batch_x, batch_y, lang, typ, dom, cor in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            lang, typ, dom, cor = lang.to(device), typ.to(device), dom.to(device), cor.to(device)
            logits = model(batch_x, lang, typ, dom)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.tolist())
            labels.extend(batch_y.tolist())
            langs.extend(lang.tolist())
            types.extend(typ.tolist())
            corpus.extend(cor.tolist())

        acc = accuracy_score(labels, preds)
        wandb.log({"epoch": epoch+1, "val_accuracy": acc})

        # Lang+Framework+Corpus
        print("Dataset:")
        joint_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for l, t, c, p, y in zip(langs, types, corpus, preds, labels):
            joint_acc[(l, t, c)]["correct"] += int(p == y)
            joint_acc[(l, t, c)]["total"] += 1

        for (lang_id, type_id, corpus_id), stat in joint_acc.items():
            lang_name = list(lang2id.keys())[list(lang2id.values()).index(lang_id)]
            type_name = list(type2id.keys())[list(type2id.values()).index(type_id)]
            corpus_name = list(corpus2id.keys())[list(corpus2id.values()).index(corpus_id)]

            acc_joint = stat["correct"] / stat["total"]
            print({f"val_acc/{lang_name}_{type_name}_{corpus_name}": acc_joint})

        # Lang
        print("Lang:")
        lang_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for l, p, y in zip(langs, preds, labels):
            lang_acc[l]["correct"] += int(p == y)
            lang_acc[l]["total"] += 1

        for lang_id, stat in lang_acc.items():
            lang_name = list(lang2id.keys())[list(lang2id.values()).index(lang_id)]
            acc_lang = stat["correct"] / stat["total"]
            print({f"val_acc/lang_{lang_name}": acc_lang})

        # Framework
        print("Framework:")
        type_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for t, p, y in zip(types, preds, labels):
            type_acc[t]["correct"] += int(p == y)
            type_acc[t]["total"] += 1

        for type_id, stat in type_acc.items():
            type_name = list(type2id.keys())[list(type2id.values()).index(type_id)]
            acc_type = stat["correct"] / stat["total"]
            print({f"val_acc/type_{type_name}": acc_type})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}")
