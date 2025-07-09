# Georgetown University Team for DISRPT 2025 Shared Task

## Setup

We use conda for maintaining a python Dev environment and `requirements.txt` for cataloguing the dependencies
1. Create (or activate) the environment.
```bash
conda create -n disrpt python==3.10
conda activate disrpt
```

2. Install dependencies
Dependencies can be installed with
```
python -m pip install -r requirements.txt
```

## Preparing Task Data
Clone the shared task public repo (or add temporarily as submodule)
Change the CLI directory to the shared task repo `sharedtask2025/util` and run `python process_underscore.py`
Once the process is complete copy the data to `/data`

The `.gitignore` is updated to not commit the data but it can change if we decide to change that

## Dependency Management
However, we only use pip for dependency management.
All the dependencies for the package must be added to the `requirements.txt`


## Run
```bash
python run_model.py --dataset_name deu.rst.pcc \
  --model_name google-bert/bert-base-multilingual-cased 
```

## Dataset Processing
We use the `datasets` library to load and train/evaluate with the dataset.
`disrptdata.py` has the code to load the dataset and process it including adding features.

For the relation classification task, the `*.rels` data are the backbone for the task. 
The `*.tok` and `*.conllu` files contain the tokens and dependency information, that can then be used to provide features.

`features.py` adapts the pre-processing of data for features from https://github.com/gucorpling/DisCoDisCo and https://gitlab.irit.fr/melodi/andiamo/relation-classification-features/

## Feature Groups
There are multiple set of features and they come from a variety of levels of the data


### Feature Group 1: Data Features
These are effectively those provided in the data i.e. feature columns of the `*.rels`
Feature Name | Description | Level
---|---|---
(s1|s2)_toks | Tokens for the Sentence with Unit 1 | Unit
dir | Direction of the relation | Relations
language | Code for the Language | Corpus
corpus | Name of the Corpus | Corpus
framework | Name of the Framework | Corpus

### Feature Group 2: Common Disco Features 
Some features from DiscoDisco were used in at least one of the models. 
These are considered `common` group of features.




## Features from Table 1 of https://aclanthology.org/2024.codi-1.9/
Here's the text formatted as a markdown table:

| Feature in JSON | Feature Type | Example | Description | Used |
|---|---|---|---|---|
| nuc_children | Nucleus' Children Num. | 2 | No. of discourse units in Unit 1 | 5 |
| sat_children | Satellite's Children Num. | 2 | No. of discourse units in Unit 2 | 8 |
| genre | Genre Cat. | reddit | Genre of a document (where available) | 5 |
| u1_discontinuous | Discontinuous Cat. | True | Whether Unit 1's tokens are not all contiguous in the text | 3 |
| u2_discontinuous | Discontinuous Cat. | True | Whether Unit 2's tokens are not all contiguous in the text | 5 |
| u1_issent | Is Sentence Cat. | True | Whether Unit 1 is a whole sentence | 3 |
| u2_issent | Is Sentence Cat. | True | Whether Unit 2 is a whole sentence | 5 |
| length_ratio | Length Ratio Num. | 0.3 | Ratio of unit 1 and unit 2's token lengths | 3 |
| same_speaker | Same Speaker Cat. | True | Whether the same speaker produced Unit 1 and Unit 2 | 2 |
| u1_func | Unit Function Cat. | root | Universal Dependencies Relation of Unit 1's Head to the Head of the input sequence | 1 |
| u1_depdir | Universal Part of speech Tag Cat. | ROOT | Part of speech of the Unit 1's Head wrt. the Head of the input sequence | 8 |
| u2_func | Unit Function Cat. | advcl | Universal Dependencies Relation of Unit 2's Head to the Head of the input sequence | 8 |
| u2_pos | Part of speech & Morphological Tag Cat. | VB | Part of speech & Morphological tag of the Unit 2's Head | 8 |
| u2_depdir | Universal Part of speech Tag Cat. | LEFT | Part of speech of the Unit 2's Head wrt. the Head of the input sequence | 7 |
| u1_position | Position Num. | 0.4 | Position of Unit 1 in the document, between 0.0 and 1.0 | 9 |
| distance | Distance Num. | 7 | No. of other discourse units between Unit 1 and Unit 2 | 9 |
| lex_overlap_length | Lexical Overlap Num. | 3 | No. of overlapping non-stoplist words in Unit 1 and Unit 2 | 1 |
| unit1_case | Uppercased letter Cat. | cap_initial | Whether the unit starts with a capital letter or not | 1 |
| unit2_case | Uppercased letter Cat. | other | Whether the unit starts with a capital letter or not | 1 |

## Feature Group 3: All
| Feature in JSON | Feature Type | Example | Description | Used |
|---|---|---|---|---|
| u1_length | Length Num. | 9 | Length of Unit 1, in tokens |  |
| u2_length | Length Num. | 13 | Length of Unit 2, in tokens |  |
| u1_speaker | Name of Speaker 1 Cat. | Rainbow | Name of Speaker (available only for STAC) |  |
| u2_speaker | Name of Speaker 2 Cat. | Markus | Name of Speaker (available only for STAC) |  |
| u1_pos | Part of speech & Morphological Tag Cat. | VBN | Part of speech & Morphological tag of the Unit 1's Head |  |
| doclen | Document Length Num. | 214 | Length of the document, in tokens |  |
| u2_position | Position Num. | 0.4 | Position of Unit 2 in the document, between 0.0 and 1.0 |  |
| percent_distance | Percent of distance Num. | 0.05 | No. of discourse units between Unit 1 and Unit 2 divided by sequence length |  |
| lex_overlap_words | Lexical Overlap Cat. | assets sold | List of overlapping non-stoplist words in Unit 1 and Unit 2 |  |