"""
In this module, we define the dataset utilities for collating, tokenizing, and preparing the datasets.
This includes reading the DISRPT data files, collating them as needed and providing them only as huggingface (torch) datasets.
"""
import io
import datasets
from datasets import Dataset, DatasetDict, load_dataset
from util import get_logger
from pathlib import Path

DATA_DIR = "data"

logger = get_logger(__name__)

def get_meta_features_for_dataset(dataset_name):
    lang, framework, source_dataset = dataset_name.split(".")
    source_dataset = source_dataset.split("_")[0]  # Remove any suffix like -v1
    return lang, framework, source_dataset

def get_list_of_dataset_from_data_dir(data_dir):
    datasets = [child.name for child in Path(data_dir).iterdir()]
    print("Found the following datasets in the data directory:")
    return datasets

def get_dataset(language, framework, dataset_key):
    return load_training_dataset(f"{language}.{framework}.{dataset_key}")

def read_rels_split(split_prefix, lang, framework, corpus, context_size=-1):
    # Ref : https://github.com/disrpt/sharedtask2025/blob/091404690ed4912ca55873616ddcaa7f26849308/utils/disrpt_eval_2024.py#L246
    data = io.open(split_prefix + ".rels", encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]
    LABEL_ID = -1
    TYPE_ID = -3
    U1_ID = 5
    U2_ID = 6
    DIRECTION_ID = -4
    match context_size:
        case -1:  # No context
            pass
        # If context size is specified, we can load it
        case x if isinstance(x, int) and x > 0:
            raise NotImplementedError("Context size support is not implemented yet.")

    labels = [line[LABEL_ID] for line in split_lines]
    u1s = [line[U1_ID] for line in split_lines]
    u2s = [line[U2_ID] for line in split_lines]
    directions = [line[DIRECTION_ID] for line in split_lines]
    types = [line[TYPE_ID] for line in split_lines]
    logger.info(f"Loading {split_prefix} with features {lang} {framework} {corpus} ")
    return Dataset.from_dict({
        "lang": [lang] * len(labels),
        "framework": [framework] * len(labels),
        "corpus": [corpus] * len(labels),
        "label": labels,
        "type": types,
        "u1": u1s,
        "u2": u2s,
        "text": [f"{u1} {u2}" for u1, u2 in zip(u1s, u2s)],
        "direction": directions,
    })

# Load Dev and Train Datasets if they are present
def load_training_dataset(dataset_name, lang, framework, corpus, context_size=-1):
    # TODO add three identification from the dataset name that can be used
    # TODO Language Token
    # TODO Dataset Token
    # TODO Task Token
    # TODO Direction Token to the input
    # TODO text context
    # not compatible with a the pandas based \t reader try with explicit code
    # return load_dataset('csv', data_files={'dev': dev_filepath, 'train': train_filepath}, delimiter='\t')
    dataset = DatasetDict()
    def load_split_if_it_exists(split_name):
        if Path(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}.rels").exists() is True:
            logger.info(f"Loading {split_name} dataset for {dataset_name}")
            rels = read_rels_split(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}",
                                   lang, framework, corpus, context_size)
            dataset[split_name] = rels
        else:
            logger.warning(f"No {split_name} split found for {dataset_name}.")
    load_split_if_it_exists("dev")
    load_split_if_it_exists("train")
    return dataset

def get_combined_dataset():
    """
    Combine all datasets into a single DatasetDict.
    """
    combined_dataset = DatasetDict()
    all_datasets = [load_training_dataset(dataset_name, *get_meta_features_for_dataset(dataset_name)) for dataset_name in get_list_of_dataset_from_data_dir(DATA_DIR)]
    combined_dataset["dev"] = datasets.concatenate_datasets(
        [dataset["dev"] for dataset in all_datasets if "dev" in dataset]
    )
    combined_dataset["train"] = datasets.concatenate_datasets(
        [dataset["train"] for dataset in all_datasets if "train" in dataset]
    )
    return combined_dataset



if __name__ == "__main__":
    # Sanity check for the dataset loading
    logger.info(get_list_of_dataset_from_data_dir("data"))
    for dataset_name in get_list_of_dataset_from_data_dir("data"):
        lang, framework, corpus = get_meta_features_for_dataset(dataset_name)
        logger.info(f"Loading dataset: {dataset_name}")
        logger.info(f"Language: {lang}, Framework: {framework}, Dataset: {corpus}")
        dataset = load_training_dataset(dataset_name, lang, framework, corpus)
        logger.info(dataset)

    combined_dataset = get_combined_dataset()
    logger.info(f"Combined Dataset: {combined_dataset}")
    # Sample 5 items from the combined dataset
    logger.info(f"Sampling from combined dataset")
    for item in combined_dataset['dev'].shuffle(seed=42).select(range(5)):
        logger.info(item)

