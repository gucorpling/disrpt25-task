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
from mpmath.matrices.matrices import rowsep

import features
from util import get_logger

DATA_DIR = "data"

logger = get_logger(__name__)

def get_meta_features_for_dataset(dataset_name):
    lang, framework, source_dataset = dataset_name.split(".")
    source_dataset = source_dataset.split("_")[0]  # Remove any suffix like -v1
    return lang, framework, source_dataset

def get_list_of_dataset_from_data_dir(data_dir):
    datasets = [child.name for child in Path(data_dir).iterdir() if child.is_dir()]
    datasets = ['ces.rst.crdt', 'deu.rst.pcc', 'eng.dep.covdtb', 'eng.dep.scidtb', 'eng.erst.gentle', 'eng.erst.gum', 'eng.pdtb.gentle', 'eng.pdtb.gum', 'eng.pdtb.pdtb', 'eng.pdtb.tedm', 'eng.rst.oll', 'eng.rst.rstdt', 'eng.rst.sts', 'eng.sdrt.msdc', 'eng.sdrt.stac', 'eus.rst.ert', 'fas.rst.prstc', 'fra.sdrt.annodis', 'ita.pdtb.luna', 'nld.rst.nldt', 'por.pdtb.crpc', 'por.pdtb.tedm', 'por.rst.cstn', 'rus.rst.rrt', 'spa.rst.rststb', 'spa.rst.sctb', 'tha.pdtb.tdtb', 'tur.pdtb.tdb', 'tur.pdtb.tedm', 'zho.dep.scidtb', 'zho.pdtb.cdtb', 'zho.rst.gcdt', 'zho.rst.sctb']
    # datasets = ['ces.erst.gum', 'deu.erst.gum', 'eus.erst.gum', 'fas.rst.rstdt', 'fra.erst.gum', 'nld.erst.gum', 'ces.rst.crdt', 'deu.rst.pcc', 'eng.dep.covdtb', 'eng.dep.scidtb', 'eng.erst.gentle', 'eng.erst.gum', 'eng.pdtb.gentle', 'eng.pdtb.gum', 'eng.pdtb.pdtb', 'eng.pdtb.tedm', 'eng.rst.oll', 'eng.rst.rstdt', 'eng.rst.sts', 'eng.sdrt.msdc', 'eng.sdrt.stac', 'eus.rst.ert', 'fas.rst.prstc', 'fra.sdrt.annodis', 'ita.pdtb.luna', 'nld.rst.nldt', 'por.pdtb.crpc', 'por.pdtb.tedm', 'por.rst.cstn', 'rus.rst.rrt', 'spa.rst.rststb', 'spa.rst.sctb', 'tha.pdtb.tdtb', 'tur.pdtb.tdb', 'tur.pdtb.tedm', 'zho.dep.scidtb', 'zho.pdtb.cdtb', 'zho.rst.gcdt', 'zho.rst.sctb']
    # datasets = ['ces.erst.gum', 'deu.erst.gum', 'eus.erst.gum', 'fas.rst.rstdt', 'fra.erst.gum', 'nld.rst.oll', 'nld.rst.sts', 'ces.rst.crdt', 'deu.rst.pcc', 'eng.dep.covdtb', 'eng.dep.scidtb', 'eng.erst.gentle', 'eng.erst.gum', 'eng.pdtb.gentle', 'eng.pdtb.gum', 'eng.pdtb.pdtb', 'eng.pdtb.tedm', 'eng.rst.oll', 'eng.rst.rstdt', 'eng.rst.sts', 'eng.sdrt.msdc', 'eng.sdrt.stac', 'eus.rst.ert', 'fas.rst.prstc', 'fra.sdrt.annodis', 'ita.pdtb.luna', 'nld.rst.nldt', 'por.pdtb.crpc', 'por.pdtb.tedm', 'por.rst.cstn', 'rus.rst.rrt', 'spa.rst.rststb', 'spa.rst.sctb', 'tha.pdtb.tdtb', 'tur.pdtb.tdb', 'tur.pdtb.tedm', 'zho.dep.scidtb', 'zho.pdtb.cdtb', 'zho.rst.gcdt', 'zho.rst.sctb']
    logger.info("Found the following datasets in the data directory:")
    return datasets

def get_dataset(dataset_name, context_sent=0, context_tok=0, include_common_features=True, include_noncommon_features=True):
    return load_training_dataset(dataset_name, context_sent, context_tok, include_common_features,
                                 include_noncommon_features)

# Read the .conllu files and convert them to a dataset
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
        sent_id = 0
        if 'newdoc id' in sentence.metadata:
            doc_id = sentence.metadata['newdoc id']
            assert doc_id not in organized_sentences, f"Duplicate doc_id found: {doc_id}"
            organized_sentences[doc_id] = []
        # sent_id = sentence.metadata['sent_id']
        # assert doc_id is not None, f"Doc_id should be set before adding sentences. At sent_id {sent_id} without doc_id"
        organized_sentences[doc_id].append(sentence)
    return organized_sentences

# Why using conllu: some tok files don't have seg information
def get_segs_and_toks_for_docs_from_conllu(split_prefix):
    data = get_conllu_sentences_by_docs(split_prefix)

    sents_for_docs = dict()  # {'file_name': [[tok1, tok2, ...], ...]}
    toks_for_docs = dict()

    for fn in data:
        sents_for_docs[fn] = []
        toks_for_docs[fn] = []
        for sentence in data[fn]:
            sent = []
            for token in sentence:
                if '-' not in str(token['id']):
                    sent.append(token['form'])
                    toks_for_docs[fn].append(token['form'])
            sents_for_docs[fn].append(sent)

    # Index for fast access
    lr2idx = {}     # {'file_name': {(l, r): idx}}
    idx2lr = {}     # {'file_name': {idx: (l, r)}}

    for fn in sents_for_docs:
        sentence_ls = sents_for_docs[fn]
        cnt = 1
        for idx, sent in enumerate(sentence_ls):
            l = cnt
            cnt += len(sent)
            r = cnt - 1
            if fn not in idx2lr:
                idx2lr[fn], lr2idx[fn] = {}, {}
            idx2lr[fn][idx] = (l, r)
            lr2idx[fn][(l, r)] = idx

    return lr2idx, idx2lr, toks_for_docs

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
    toks_for_docs = dict()

    if with_segs is True:
        for doc in docs:
            toks = []
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
                            toks.append(token['form'])

                        case 'O':
                            # tur.pdtb.tdb may not have 0 tags before B
                            if span is None:
                                span = list()
                            span.append(token['form'])
                            toks.append(token['form'])

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
            toks_for_docs[doc.metadata['newdoc id']] = toks

    # Forget to handle MWTs
    # toks_for_docs = {
    #     doc.metadata['newdoc id']: [token['form'] for token in doc] for doc in docs
    # }
    total_spans = sum(len(doc_segs) for doc_segs in segs_for_docs.values())
    total_tokens = sum(len(tokens) for tokens in toks_for_docs.values())
    logger.info(f"Loaded {len(docs)} documents with {total_tokens} tokens and {total_spans} segments for {split_prefix}.")
    return segs_for_docs, toks_for_docs

# Context: [pre_context, full sentence s1+s2, post_context]
def get_context(s1ors2, doc_id, s_toks, lr2idx, idx2lr, toks_for_docs, context_sent, context_tok):
    lr2idx = lr2idx[doc_id]
    idx2lr = idx2lr[doc_id]
    toks_for_docs = toks_for_docs[doc_id]

    # eng.pdtb.tedm talk_1927_en line 52 614-623,624-715
    if "," in s_toks:
        ranges = s_toks.split(",")
        s_start, s_end = None, None

        for r in ranges:
            s, e = map(int, r.split('-'))

            if s_start is None or s < s_start:
                s_start = s
            if s_end is None or e > s_end:
                s_end = e

    elif "-" in s_toks:
        s_start, s_end = s_toks.split("-")
    else:
        s_start = s_end = s_toks

    if (int(s_start), int(s_end)) in lr2idx:
        
        s = ' '.join(toks_for_docs[(int(s_start)-1):int(s_end)])
        idx = lr2idx[(int(s_start), int(s_end))]
    else:
        # example: eng.rst.rstdt_dev wsj_0629 (942, 1042)
        sentence_range = []

        for (start, end), sentence_number in lr2idx.items():
            if not (int(s_end) < start or int(s_start) > end):
                sentence_range.append(sentence_number)

        s = ' '.join(' '.join(toks_for_docs[s_start-1:s_end]) for s_r in sentence_range for s_start, s_end in [idx2lr[s_r]])
        idx = sentence_range[0] if s1ors2 == 1 else sentence_range[-1]
        
    context_idx = idx
    context = []
    while abs(idx - context_idx) < context_sent or sum(len(sublist) for sublist in context) < context_tok:
        context_idx = context_idx - 1 if s1ors2 == 1 else context_idx + 1
        if context_idx not in idx2lr:
            break
        lr = idx2lr[context_idx]
        context.insert(0, toks_for_docs[lr[0]-1:lr[1]]) if s1ors2 == 1 else context.append(toks_for_docs[lr[0]-1:lr[1]])
    return s, " ".join(word for sublist in context for word in sublist)

def read_rels_split(split_prefix, lang, framework, corpus, context_sent, context_tok):
    # Ref : https://github.com/disrpt/sharedtask2025/blob/091404690ed4912ca55873616ddcaa7f26849308/utils/disrpt_eval_2024.py#L246
    data = io.open(split_prefix + ".rels", encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    # TODO optimize collation?
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]
    # TODO make the headers enum with indices so you can just use the name
    DIRECTION_ID = -4
    TYPE_ID = -3
    LABEL_ID = -1
    UNIT_1_TOKS = 2
    UNIT_2_TOKS = 3
    U1_ID = 5
    U2_ID = 6
    S1_TOKS = 7
    S2_TOKS = 8
    S1_SENT_ID = 9
    S2_SENT_ID = 10
    DIRECTION_ID = -4

    labels = [line[LABEL_ID] for line in split_lines]
    doc_ids = [line[0] for line in split_lines]
    u1s = [line[U1_ID] for line in split_lines]
    u1_toks = [line[UNIT_1_TOKS].split("-") for line in split_lines]
    u2s = [line[U2_ID] for line in split_lines]
    u2_toks = [line[UNIT_2_TOKS].split("-") for line in split_lines]
    u1_sents = [line[S1_SENT_ID] for line in split_lines]
    u2_sents = [line[S2_SENT_ID] for line in split_lines]
    directions = [line[DIRECTION_ID] for line in split_lines]
    types = [line[TYPE_ID] for line in split_lines]
    s1_toks = [line[S1_TOKS] for line in split_lines]
    s2_toks = [line[S2_TOKS] for line in split_lines]

    logger.info(f"Loading {split_prefix} with features {lang} {framework} {corpus} ")

    lr2idx, idx2lr, toks_for_docs = get_segs_and_toks_for_docs_from_conllu(split_prefix)
    contexts = []
    if context_sent > 0 or context_tok > 0:
        for i in range(0, len(split_lines)):
            doc_id = doc_ids[i]
            s1_tok = s1_toks[i]
            s2_tok = s2_toks[i]
            s1, s1_context = get_context(1, doc_id, s1_tok, lr2idx, idx2lr, toks_for_docs, context_sent, context_tok)
            s2, s2_context = get_context(2, doc_id, s2_tok, lr2idx, idx2lr, toks_for_docs, context_sent, context_tok)

            if s1_tok == s2_tok:
                context = [s1_context, s1, s2_context]
            else:
                context = [s1_context, s1+s2, s2_context]

            contexts.append(context)

    data_dict = {
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
        "u1_sent": u1_sents,
        "u2_sent": u2_sents,
        "text": [f"{u1} {u2}" for u1, u2 in zip(u1s, u2s)],
        "direction": directions,
    }
    if len(contexts) > 0:
        data_dict["context"] = contexts

    return Dataset.from_dict(data_dict)

# Load Dev and Train Datasets if they are present
# context_sent: control the number of sentences; context_tok: control the number of tokens. Satisfy both of them
def load_training_dataset(dataset_name, sentences_for_context, tokens_for_context, include_common_features,
                          include_noncommon_features):
    # TODO add three identification from the dataset name that can be used
    # not compatible with a the pandas based \t reader try with explicit code
    # return load_dataset('csv', data_files={'dev': dev_filepath, 'train': train_filepath}, delimiter='\t')
    dataset = DatasetDict()
    # Process relfiles to create a dataset with features
    dataset_prefix = f"{DATA_DIR}/{dataset_name}/"
    lang, framework, corpus = get_meta_features_for_dataset(dataset_name)

    def load_split_if_it_exists(split_name):
        rels_file = f"{dataset_prefix}{dataset_name}_{split_name}.rels"
        if Path(rels_file).exists() is True:
            conllu_file = f"{dataset_prefix}{dataset_name}_{split_name}.conllu"
            # TODO reduce double reading and use the featured dataset as the prime source
            rels = read_rels_split(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}", lang, framework, corpus, sentences_for_context, tokens_for_context)
            # spans, tokens = get_segs_and_toks_for_docs(f"{DATA_DIR}/{dataset_name}/{dataset_name}_{split_name}", with_segs=True)
            # Just splicing the rows_with_features to the rels
            if include_common_features is True or include_noncommon_features is True:
                rows_with_features = features.process_relfile(rels_file, conllu_file, dataset_name)

            common_feat_keys = ['nuc_children', 'sat_children', 'genre', 'u1_discontinuous', 'u2_discontinuous',
                                'u1_issent', 'u2_issent', 'length_ratio', 'same_speaker', 'u1_func', 'u1_depdir',
                                'u2_func', 'u2_pos', 'u2_depdir', 'u1_position', 'distance', 'lex_overlap_length',
                                'unit1_case', 'unit2_case']

            noncommon_feat_keys = ['u1_length', 'u2_length', 'u1_speaker', 'u2_speaker', 'u1_pos', 'doclen',
                                   'u2_position', 'percent_distance', 'lex_overlap_words']

            if include_common_features is True:
                rels = rels.map(lambda x, i:  {feat:rows_with_features[i][feat] for feat in common_feat_keys} , with_indices=True)

            if include_noncommon_features is True:
                rels = rels.map(lambda x, i:  {feat:rows_with_features[i][feat] for feat in noncommon_feat_keys} , with_indices=True)

            # rels = rels.map(
            #     lambda x: {
            #         "doc_tokens": tokens[x["doc_id"]],
            #     }
            # )

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

def get_combined_dataset(context_sent=0, context_tok=0, include_common_features=False, include_noncommon_features=False):
    """
    Combine all datasets into a single DatasetDict.
    """
    combined_dataset = DatasetDict()
    all_datasets = [get_dataset(dataset_name, context_sent, context_tok, include_common_features, include_noncommon_features) for dataset_name in get_list_of_dataset_from_data_dir(DATA_DIR)]

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
                dataset = get_dataset(dataset_name, 0, 0, False, True)
                logger.info(dataset)

            combined_dataset = get_combined_dataset(context_sent=0, include_common_features=True)

            # Sample 5 items from the combined dataset
            logger.info(f"Sampling from combined dataset")
            for item in combined_dataset['dev'].shuffle(seed=42).select(range(5)):
                logger.info(item)

        case dataset_name:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = get_dataset(dataset_name)