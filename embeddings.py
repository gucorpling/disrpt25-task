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
framework_list = sorted(set(example["framework"] for example in dataset["train"]))
direction_list = sorted(set(example["direction"] for example in dataset["train"]))
corpus_list = sorted(set(
    example["corpus"] for split in ["train", "dev"]
    for example in dataset[split]
)) # only for eval

lang2id = {lang: idx for idx, lang in enumerate(lang_list)}
framework2id = {frame: idx for idx, frame in enumerate(framework_list)}
direction2id = {d: idx for idx, d in enumerate(direction_list)}
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

def prepare_embeddings_and_features(hf_dataset, label2id, lang2id, framework2id, direction2id, corpus2id, split_name, embed_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fx = f"embeddings/embeddings_{split_name}_x.pt"
    fy = f"embeddings/embeddings_{split_name}_y.pt"
    flang = f"embeddings/embeddings_{split_name}_lang.pt"
    fframework = f"embeddings/embeddings_{split_name}_framework.pt"
    fdirec = f"embeddings/embeddings_{split_name}_direction.pt"
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

    frame = load_or_generate_tensor(
        fframework, "framework features",
        lambda: torch.tensor([framework2id[ex["framework"]] for ex in hf_dataset])
    )

    direc = load_or_generate_tensor(
        fdirec, "direction features",
        lambda: torch.tensor([direction2id[ex["direction"]] for ex in hf_dataset])
    )

    cor = load_or_generate_tensor(
        fcor, "corpus features",
        lambda: torch.tensor([corpus2id[ex["corpus"]] for ex in hf_dataset])
    )

    return x, y, lang, frame, direc, cor

train_x, train_y, train_langs, train_framework, train_direction, train_corpus=prepare_embeddings_and_features(
    dataset["train"], label2id, lang2id, framework2id, direction2id, corpus2id, "train", embed_model)
val_x, val_y, val_langs, val_framework, val_direction, val_corpus=prepare_embeddings_and_features(
    dataset["dev"], label2id, lang2id, framework2id, direction2id, corpus2id, "dev", embed_model)

train_dataset = TensorDataset(train_x, train_y, train_langs, train_framework, train_direction, train_corpus)
val_dataset = TensorDataset(val_x, val_y, val_langs, val_framework, val_direction, val_corpus)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, lang_size, framework_size, direction_size, corpus_size):
        super().__init__()
        self.lang_embedding = nn.Embedding(lang_size, 16)
        self.framework_embedding = nn.Embedding(framework_size, 16)
        self.direction_embedding = nn.Embedding(directionn_size, 16)
        self.fc1 = nn.Linear(input_dim + 48, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lang, frame, direc):
        lang_embed = self.lang_embedding(lang)
        framework_embed = self.framework_embedding(frame)
        direc_embed = self.direction_embedding(direc)
        enhanced = torch.cat([x, lang_embed, framework_embed, direc_embed], dim=-1)
        return self.fc2(self.dropout(self.relu(self.fc1(enhanced))))

model = Classifier(input_dim, hidden_dim, num_classes,
                   lang_size=len(lang2id), framework_size=len(framework2id), direction_size=len(direction2id), corpus_size=len(corpus2id)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_x, batch_y, lang, frame, direc, cor in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        lang, frame, direc = lang.to(device), frame.to(device), direc.to(device)
        logits = model(batch_x, lang, frame, direc)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    wandb.log({"epoch": epoch+1, "train_loss": total_loss})

    model.eval()
    preds, labels, langs, frameworks, corpus= [], [], [], [], []

    with torch.no_grad():
        for batch_x, batch_y, lang, frame, direc, cor in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            lang, frame, direc, cor = lang.to(device), frame.to(device), direc.to(device), cor.to(device)
            logits = model(batch_x, lang, frame, direc)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.tolist())
            labels.extend(batch_y.tolist())
            langs.extend(lang.tolist())
            frameworks.extend(frame.tolist())
            corpus.extend(cor.tolist())

        acc = accuracy_score(labels, preds)
        wandb.log({"epoch": epoch+1, "val_accuracy": acc})

        # Lang+Framework+Corpus
        print("Dataset:")
        joint_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for l, t, c, p, y in zip(langs, frameworks, corpus, preds, labels):
            joint_acc[(l, t, c)]["correct"] += int(p == y)
            joint_acc[(l, t, c)]["total"] += 1

        for (lang_id, framework_id, corpus_id), stat in joint_acc.items():
            lang_name = list(lang2id.keys())[list(lang2id.values()).index(lang_id)]
            framework_name = list(framework2id.keys())[list(framework2id.values()).index(framework_id)]
            corpus_name = list(corpus2id.keys())[list(corpus2id.values()).index(corpus_id)]

            acc_joint = stat["correct"] / stat["total"]
            print({f"val_acc/{lang_name}_{framework_name}_{corpus_name}": acc_joint})

        # # Lang
        # print("Lang:")
        # lang_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        # for l, p, y in zip(langs, preds, labels):
        #     lang_acc[l]["correct"] += int(p == y)
        #     lang_acc[l]["total"] += 1

        # for lang_id, stat in lang_acc.items():
        #     lang_name = list(lang2id.keys())[list(lang2id.values()).index(lang_id)]
        #     acc_lang = stat["correct"] / stat["total"]
        #     print({f"val_acc/lang_{lang_name}": acc_lang})

        # # Framework
        # print("Framework:")
        # type_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        # for t, p, y in zip(types, preds, labels):
        #     type_acc[t]["correct"] += int(p == y)
        #     type_acc[t]["total"] += 1

        # for type_id, stat in type_acc.items():
        #     type_name = list(type2id.keys())[list(type2id.values()).index(type_id)]
        #     acc_type = stat["correct"] / stat["total"]
        #     print({f"val_acc/type_{type_name}": acc_type})

        # avg_loss = total_loss / len(train_loader)
        # print(f"Epoch {epoch+1}: Avg Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}")
