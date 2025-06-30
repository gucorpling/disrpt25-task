"""
In this module, we define the dataset utilities for collating, tokenizing, and preparing the datasets.
This includes reading the DISRPT data files, collating them as needed and providing them only as huggingface (torch) datasets.
"""
import argparse
import io
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict, load_dataset
import conllu

from util import get_logger

DATA_DIR = "data"

logger = get_logger(__name__)

def get_meta_features_for_dataset(dataset_name):
    lang, framework, source_dataset = dataset_name.split(".")
    source_dataset = source_dataset.split("_")[0]  # Remove any suffix like -v1
    return lang, framework, source_dataset

def get_list_of_dataset_from_data_dir(data_dir):
    datasets = [child.name for child in Path(data_dir).iterdir()]
    logger.info("Found the following datasets in the data directory:")
    return datasets

def get_dataset(dataset_name, context_size=-1):
    language, framework, corpus = get_meta_features_for_dataset(dataset_name)
    return load_training_dataset(dataset_name, language, framework, corpus, context_size=context_size)

# Read the .conllu files and convert them to a dataset
def read_conll_split(split_prefix):
    # Ref : ABCD
    conll_file = split_prefix + ".conllu"
    if not Path(conll_file).exists():
        raise FileNotFoundError(f"Conll file {conll_file} does not exist.")

def read_conll_sentences_split(filepath):
    # Ref : ABCD
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Conll file {filepath} does not exist.")

    with open(filepath, encoding="utf-8") as f:
        text = f.read()
        conllu_sentences = conllu.parse(text)
    return conllu_sentences

# Organize the conllu sentences into a index with dictionary using doc_id from the metadata as key
def get_conllu_sentences_by_docs(split_prefix):
    """
    Organizes conllu sentences into a dictionary indexed by doc_id.
    Each entry in the dictionary contains the sentences for that doc_id.
    """
    # Read the .conllu/tok files and convert them to a dataset
    filepath = split_prefix + ".conllu"
    conllu_sentences = read_conll_sentences_split(filepath)
    organized_sentences = {}
    doc_id = None  # Initialize doc_id to None
    for sentence in conllu_sentences:
        if 'newdoc id' in sentence.metadata:
            doc_id = sentence.metadata['newdoc id']
            assert doc_id not in organized_sentences, f"Duplicate doc_id found: {doc_id}"
            organized_sentences[doc_id] = []
        sent_id = sentence.metadata['sent_id']
        assert doc_id is not None, f"Doc_id should be set before adding sentences. At sent_id {sent_id} without doc_id"
        organized_sentences[doc_id].append(sentence)
    return organized_sentences

# Read the .tok files and organize them by doc
def get_segs_and_toks_for_docs(split_prefix, with_segs=False):
    """
    Reads the .tok files and organizes them into a dictionary indexed by doc_id.
    Each entry in the dictionary contains the sentences for that doc_id.
    """
    # Read the .conllu/tok files and convert them to a dataset
    filepath = split_prefix + ".tok"
    docs = read_conll_sentences_split(filepath)

    def bio_tag(token):
        # Neglect MWTs
        # Neglect connective tags
        # Should be only because it is a dot token
        # TODO move this decision logic to be at dataset level based on dataset config

        if 'misc' not in token:
            return None
        elif token['misc'] is None or len(token['misc']) == 0:
            return None
        elif 'Conn' in token['misc']:
            return token['misc']['Conn']
        elif 'Seg' in token['misc']:
            return token['misc']['Seg']

    # Use the BIO encodings to organize the spans by the sentences
    # Convert doc to two sequences of tokens and BIO tags

    segs_for_docs = dict()

    if with_segs is True:
        for doc in docs:
            segs = []
            span = None
            for token in doc:
                try:
                    tag = bio_tag(token)

                    match tag:
                        case None:
                            # Skip MWTs
                            continue

                        case 'B-seg' | 'B-Conn':
                            if span is not None:
                                segs.append(span)
                            span = [token['form']]

                        case 'O':
                            # tur.pdtb.tdb may not have 0 tags before B
                            if span is None:
                                span = list()
                            span.append(token['form'])

                except Exception as e:
                    logger.error(f"Error processing token {token} in doc {doc.metadata['newdoc id']}: {e}")
                    logger.error(f"Token keys: {token.keys()}")
                    for key in token.keys():
                        logger.error(f"Token {key}: {token[key]}")
                    raise e
            else:
                # Final segment without a new B tag
                if span is not None:
                    segs.append(span)

            logger.info(f"Doc {doc.metadata['newdoc id']} has {len(segs)} spans")
            segs_for_docs[doc.metadata['newdoc id']] = segs

    toks_for_docs = {
        doc.metadata['newdoc id']: [token['form'] for token in doc] for doc in docs
    }
    total_spans = sum(len(doc_segs) for doc_segs in segs_for_docs.values())
    total_tokens = sum(len(tokens) for tokens in toks_for_docs.values())
    logger.info(f"Loaded {len(docs)} documents with {total_tokens} tokens and {total_spans} segments for {split_prefix}.")
    return segs_for_docs, toks_for_docs

def read_rels_split(split_prefix, lang, framework, corpus):
    # Ref : https://github.com/disrpt/sharedtask2025/blob/091404690ed4912ca55873616ddcaa7f26849308/utils/disrpt_eval_2024.py#L246
    data = io.open(split_prefix + ".rels", encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]
    LABEL_ID = -1
    UNIT_1_TOKS = 2
    UNIT_2_TOKS = 3
    TYPE_ID = -3
    U1_ID = 5
    U2_ID = 6
    DIRECTION_ID = -4

    labels = [line[LABEL_ID] for line in split_lines]
    doc_ids = [line[0] for line in split_lines]
    u1s = [line[U1_ID] for line in split_lines]
    u1_toks = [line[UNIT_1_TOKS].split("-") for line in split_lines]
    u2s = [line[U2_ID] for line in split_lines]
    u2_toks = [line[UNIT_2_TOKS].split("-") for line in split_lines]
    directions = [line[DIRECTION_ID] for line in split_lines]
    types = [line[TYPE_ID] for line in split_lines]
    logger.info(f"Loading {split_prefix} with features {lang} {framework} {corpus} ")
    return Dataset.from_dict({
        "lang": [lang] * len(labels),
        "framework": [framework] * len(labels),
        "corpus": [corpus] * len(labels),
        "label": labels,
        "type": types,
        "doc_id": doc_ids,
        "u1": u1s,
        "u1_toks": u1_toks,
        "u2": u2s,
        "u2_toks": u2_toks,
        "text": [f"{u1} {u2}" for u1, u2 in zip(u1s, u2s)],
        "direction": directions,
    })

# Load Dev and Train Datasets if they are present
def load_training_dataset(dataset_name, lang, framework, corpus, context_size=-1):
    # TODO add three identification from the dataset name that can be used
    # TODO text context
    # not compatible with a the pandas based \t reader try with explicit code
    # return load_dataset('csv', data_files={'dev': dev_filepath, 'train': train_filepath}, delimiter='\t')
    dataset = DatasetDict()
    def load_split_if_it_exists(split_name, context_size=-1):
        if Path(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}.rels").exists() is True:
            logger.info(f"Loading {split_name} dataset for {dataset_name}")
            rels = read_rels_split(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}", lang, framework, corpus)
            spans, tokens = get_segs_and_toks_for_docs(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}", with_segs=True)
            rels = rels.map(
                lambda x: {
                    "doc_tokens": tokens[x["doc_id"]],
                }
            )

            dataset[split_name] = rels
        else:
            logger.warning(f"No {split_name} split found for {dataset_name}.")

    load_split_if_it_exists("dev")
    load_split_if_it_exists("train")

    # Augment the dataset with meta features
    if "dev" in dataset:
        pass

    if "train" in dataset:
        pass
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
    arg_parser = argparse.ArgumentParser(description="DISRPT Data Loader")
    arg_parser.add_argument("--dataset_name", type=str, default="combined", help="Name of the dataset to load. Use 'combined' to load all datasets.")

    args = arg_parser.parse_args()
    match args.dataset_name:
        case "combined":

            # Sanity check for the dataset loading
            logger.info(get_list_of_dataset_from_data_dir("data"))
            for dataset_name in get_list_of_dataset_from_data_dir("data"):
                logger.info(f"Loading dataset: {dataset_name}")
                dataset = get_dataset(dataset_name)
                logger.info(dataset)

            combined_dataset = get_combined_dataset()
            logger.info(f"Combined Dataset: {combined_dataset}")
            # Sample 5 items from the combined dataset
            logger.info(f"Sampling from combined dataset")
            for item in combined_dataset['dev'].shuffle(seed=42).select(range(5)):
                logger.info(item)

        case dataset_name:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)


