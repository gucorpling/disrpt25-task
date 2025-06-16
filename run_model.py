import argparse
import io
import json

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from pprint import pprint

from transformers import AutoModel, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments
from datasets import Dataset, load_dataset, DatasetDict
import evaluate

def get_disrpt_labels():
    """ Returns a dictionary of labels for the DISRPT task by retrieiving it from mapping_disrpt.json file."""
    mapping_file = "mapping_disrpt25.json"
    if not Path(mapping_file).exists():
        raise FileNotFoundError(f"Mapping file {mapping_file} not found. Please download it from the DISRPT repository.")
    mappings = json.load(open(mapping_file, 'r', encoding='utf-8'))
    unique_labels = set(mappings.values())
    return unique_labels

# Ref : https://github.com/disrpt/sharedtask2025/blob/091404690ed4912ca55873616ddcaa7f26849308/utils/disrpt_eval_2024.py#L246
def load_rels_dataset(dev_filepath, train_filepath):
    # TODO add three identification from the dataset name that can be used
    # TODO Language Token
    # TODO Dataset Token
    # TODO Task Token
    # TODO Direction Token to the input
    # TODO text backround
    # not compatible with a the pandas based \t reader try with explicit code
    #
    # return load_dataset('csv', data_files={'dev': dev_filepath, 'train': train_filepath}, delimiter='\t')
    def get_dataset(filepath):
        data = io.open(filepath, encoding="utf-8").read().strip().replace("\r", "")
        lines = data.split("\n")
        header = lines[0]
        split_lines = [line.split("\t") for line in lines[1:]]
        LABEL_ID = -1
        TYPE_ID = -3
        U1_ID = 5
        U2_ID = 6
        DIRECTION_ID = -4

        labels = [line[LABEL_ID] for line in split_lines]
        u1s = [line[U1_ID] for line in split_lines]
        u2s = [line[U2_ID] for line in split_lines]
        directions = [line[DIRECTION_ID] for line in split_lines]
        types = [line[TYPE_ID] for line in split_lines]

        return Dataset.from_dict({
            "label": labels,
            "type": types,
            "u1": u1s,
            "u2": u2s,
            "text": [f"{u1} {u2}" for u1, u2 in zip(u1s, u2s)],
            "direction": directions,
        })


    dev_rels = get_dataset(dev_filepath)
    train_rels = get_dataset(train_filepath)
    print(dev_rels[0])
    print(train_rels[0])
    dataset = DatasetDict()
    dataset['dev'] = dev_rels
    dataset['train'] = train_rels
    return dataset

# Fix doesn't work yet
# TODO pab make it a pipeline
# Ref: https://gitlab.irit.fr/melodi/andiamo/discret-zero-shot/-/blob/master/classifier_pytorch.py
class TransformerClassifier(nn.Module):
    def __init__(self, num_labels = 17, model_name='google-bert/bert-base-multilingual-cased'):
        super(TransformerClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_id, mask):
        """Only trains the final linear layer"""
        outputs = self.model(input_ids=input_id, attention_mask=mask)
        outputs = self.linear(outputs.last_hidden_state)
        return outputs


def train(model_name, dev_dataset, train_dataset):
    """The function uses huggingface to train model with dataset"""

    def compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        return {
            'accuracy': accuracy.compute(predictions=predictions, references=labels),
            'precision': precision.compute(predictions=predictions, references=labels, average='weighted'),
            'recall': recall.compute(predictions=predictions, references=labels, average='weighted'),
            'f1': f1.compute(predictions=predictions, references=labels, average='weighted')
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dev_labels = set(dev_dataset['label'])
    train_labels = set(train_dataset['label'])
    disrpt_labels = get_disrpt_labels()
    print("Unique labels in dev dataset:", dev_labels)
    print("Unique labels in train dataset:", train_labels)
    assert dev_labels.union(train_labels).issubset(disrpt_labels), "Labels in dev or train dataset are not in the unique labels set."

    label2id = {label: i for i, label in enumerate(sorted(disrpt_labels))}
    id2label = {i: label for label, i in label2id.items()}
    print("Num of labels:", len(label2id))

    # Tokenize the dataset for the model
    dev_dataset = dev_dataset.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=512),
        batched=True,
        remove_columns=['text', 'u1', 'u2', 'type', 'direction']
    )
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=512),
        batched=True,
        remove_columns=['text', 'u1', 'u2', 'type', 'direction']
    )
    # Encode labels as integers
    dev_dataset = dev_dataset.map(lambda x: {'label': label2id[x['label']]}, remove_columns=['label'])
    train_dataset = train_dataset.map(lambda x: {'label': label2id[x['label']]}, remove_columns=['label'])
    # # Try with smaller set
    # dev_dataset = dev_dataset.shuffle(seed=42).select(range(64))
    # train_dataset = train_dataset.shuffle(seed=42).select(range(64))

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=17, label2id=label2id, id2label=id2label)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate(dev_dataset)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=False)
    parser.add_argument("--model_name", type=str, required=True,)
    args = parser.parse_args()
    dataset = load_rels_dataset(args.dev, args.train)
    train(model_name=args.model_name, dev_dataset=dataset['dev'], train_dataset=dataset['train'])
    print(dataset)


