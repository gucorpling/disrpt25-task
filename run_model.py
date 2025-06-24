import argparse
import json

from torch import nn
from pathlib import Path

from transformers import AutoModel, Trainer, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, \
    TrainingArguments, set_seed
from datasets import Dataset, load_dataset, DatasetDict
import evaluate

import disrptdata
import util

logger = util.get_logger(__name__)
def get_disrpt_labels():
    """ Returns a dictionary of labels for the DISRPT task by retrieiving it from mapping_disrpt.json file."""
    mapping_file = "mapping_disrpt25.json"
    if not Path(mapping_file).exists():
        raise FileNotFoundError(f"Mapping file {mapping_file} not found. Please download it from the DISRPT repository.")
    mappings = json.load(open(mapping_file, 'r', encoding='utf-8'))
    unique_labels = set(mappings.values())
    return unique_labels

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
        output_dir=f"./results/{model_name}-{seed}",
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
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, action='store', default='combined',)
    parser.add_argument("--model_name", type=str, required=True,)
    args = parser.parse_args()
    # TODO support seed as a CLI parameter
    seed = 42
    set_seed(seed)

    match args.dataset_name:
        case 'combined':
            dataset = disrptdata.get_combined_dataset()

        case dataset_name:
            # if dataset_name not in disrptdata.get_list_of_dataset_from_data_dir(disrptdata.DATA_DIR):
            #     logger.error(f"Dataset {dataset_name} not found in {disrptdata.DATA_DIR}.")
            #     raise ValueError(f"Dataset {dataset_name} not found in {disrptdata.DATA_DIR}.")
            # else:
            lang, framework, corpus = disrptdata.get_meta_features_for_dataset(dataset_name)
            logger.info(f"Loading dataset: {dataset_name}")
            logger.info(f"Language: {lang}, Framework: {framework}, Dataset: {corpus}")
            dataset = disrptdata.load_training_dataset(dataset_name, lang, framework, corpus)

    print(dataset)
    results = train(model_name=args.model_name, dev_dataset=dataset['dev'], train_dataset=dataset['train'])
    print(f"Results for model {args.model_name}: {results}")
    # Save the results to a file as JSON
    output_file = f"./results/{args.model_name}/results.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


