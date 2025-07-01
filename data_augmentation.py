import disrptdata
from util import get_logger
from openai import OpenAI
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import time, io, json, re, stanza, shutil

MODEL = "gpt-4.1"
DATA_DIR = "data"
LANGUAGES = {"ces": "Czech", "deu": "German"}
LANGUAGES_STANZA = {"ces": "cs", "deu": "de"}

logger = get_logger('gpt4')

def check_string_format(s):
    pattern = r'^\{\{.+\}\}$'
    
    if re.match(pattern, s.strip()):
        return True
    return False


def call_api(model, system, message, num):
    
    client = OpenAI(api_key="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")

    tries = 0

    while True:
        try:
            # print(f"running {num}")

            response = client.chat.completions.create(
                model= model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                stream=False,
                timeout=180,
                temperature = 0.0
            )

            content = response.choices[0].message.content

            if not check_string_format(content):
                raise Exception
            
            return content
        
        except Exception as e:
            if tries > 20:
                logger.info(f"{num} Not succeed!")
                return None
            else:
                tries += 1 
                time.sleep(2)

def get_original_text(dir_name):    
    data = io.open((dir_name + ".rels"), encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    split_lines = [line.split("\t") for line in lines[1:]]

    sentence_ids = defaultdict(set)

    for line in split_lines:
        doc = line[0]
    
        for i in range(1, 3):
            sentence_ids[doc].update(line[i].split(','))
        
    return sentence_ids

def translate_original_text_into_target_language(target_lang, sentence_ids, split_name):
    output_file = f"{split_name}_{target_lang}.json"
    lang = LANGUAGES[target_lang]

    system = """## Role and Goal:
You are a translator, translate a sentence based on the context into {{LANGUAGE}} language directly without explanation.

## Guidelines:
1. You will be given a paragraph and a sentence, which is part of the paragraph. Based on the full content of the paragraph, please translate only that specific sentence, even if the sentence itself lacks completeness.
2. Only translate the sentence provided, do not translate the entire paragraph.
3. You should include all necessary punctuation in the original sentence.
4. Do not add any extraneous information.

## Output format:
Please enclose the result within double curly braces {{}}.
Example: {{Translation Result}}"""

    message_template = """## Paragraph:
{{PARAGRAPH}}
    
## Sentence need to be translated:
{{SENTENCE}}"""
    
    system =system.replace("{{LANGUAGE}}", lang)

    data = disrptdata.get_spans_and_toks_for_docs(split_name, True)[0]

    translations = defaultdict(dict)

    for doc in sentence_ids:
        doc_content = [span for sublist in data[doc] for span in sublist]
        print(f"Begin {doc}:")
        for s_i in tqdm(sentence_ids[doc]):
            if "-" not in s_i:
                sent_start = s_i
                sent_end = s_i
            else:
                sent_start, sent_end = s_i.split("-")

            sent_start = int(sent_start)
            sent_end = int(sent_end)

            para_start = max(sent_start-30, 1)
            para_end = min(sent_end+30, len(doc_content))
            paragraph = ' '.join(doc_content[(para_start-1):(para_end)])
            sentence = ' '. join(doc_content[(sent_start-1):(sent_end)])

            message = message_template.replace("{{PARAGRAPH}}", paragraph).replace("{{SENTENCE}}", sentence)

            translation = call_api(MODEL, system, message, f"{doc}_s_i")

            translations[doc][s_i] = translation

        with open(f"{split_name}_{target_lang}.json", 'w') as json_file:
            json.dump(dict(translations), json_file, indent=4)

def remap_spans(input_str, mapping):
    """Remap original span to new span using mapping dictionary."""
    result = []
    for part in input_str.split(","):
        if part in mapping:
            mapped = mapping[part]
            if "-" in mapped:
                start, end = mapped.split("-")
                result.append(start if start == end else mapped)
            else:
                result.append(mapped)
        else:
            raise ValueError(f"Span '{part}' not found in mapping.")
    return ",".join(result)

def get_translate_spans(span_str, token_dict):
    """Join tokens from token_dict for each span, and combine multiple spans with <*>."""
    parts = []
    for span in span_str.split(","):
        if span not in token_dict:
            raise ValueError(f"Span '{span}' not found in token_dict.")
        sentence = " ".join(token_dict[span])
        parts.append(sentence)
    return " <*> ".join(parts)

def tokenize_translation(data, target_lang):
    """Tokenize all translation entries using Stanza."""
    stanza_lang = LANGUAGES_STANZA[target_lang]
    stanza.download(stanza_lang)
    nlp = stanza.Pipeline(stanza_lang)

    for doc in data:
        for sentence_id in tqdm(data[doc]):
            sentence = re.search(r'\{\{(.*?)\}\}', data[doc][sentence_id]).group(1)
            sentence = nlp(sentence)
            tokens = [word.text for s in sentence.sentences for word in s.words]
            data[doc][sentence_id] = tokens
    return data

def generate_mapping(data):
    """Create new span mapping and reordered data."""
    reordered = {}
    span_mapping = {}

    for filename, spans in data.items():
        sorted_items = sorted(spans.items(), key=lambda item: int(item[0].split("-")[0]))

        new_spans = {}
        mapping = {}
        current_token_id = 1

        for original_range, tokens in sorted_items:
            span_len = len(tokens)
            new_range = f"{current_token_id}-{current_token_id + span_len - 1}"
            new_spans[new_range] = tokens
            mapping[original_range] = new_range
            current_token_id += span_len

        reordered[filename] = {k: v for k, v in sorted_items}
        span_mapping[filename] = mapping

    return reordered, span_mapping

def save_tok_file(data, output_path):
    with open(output_path, "w", encoding="utf-8") as out_tok:
        for file_id, span_dict in data.items():
            out_tok.write(f"# newdoc id = {file_id}\n")
            token_id = 1
            for span in span_dict.values():
                for i, token in enumerate(span):
                    seg_label = "B-seg" if i == 0 else "O"
                    out_tok.write(f"{token_id}\t{token}\t_\t_\t_\t_\t_\t_\t_\tSeg={seg_label}\n")
                    token_id += 1
            out_tok.write("\n")

def save_rels_file(split_name, mapping, translate_data, output_path):
    data = io.open(f"{split_name}.rels", encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]

    for line in split_lines:
        doc = line[0]
        line[1] = remap_spans(line[1], mapping[doc])
        line[2] = remap_spans(line[2], mapping[doc])
        line[3] = line[4] = ""
        line[5] = get_translate_spans(line[1], translate_data[doc])
        line[6] = get_translate_spans(line[2], translate_data[doc])
        line[7:11] = ["", "", "", ""]

    modified_lines = ["\t".join(map(str, line)) for line in split_lines]
    final_output = [header] + modified_lines

    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))

def transform_to_dataset(target_lang, split_name, output_name):
    file_name = f"{split_name}_{target_lang}.json"

    with open(file_name, 'r') as f:
        data = json.load(f)

    data = tokenize_translation(data, target_lang)

    # Save tokenized version temporarily
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    reordered, span_mapping = generate_mapping(data)

    # Save reordered translation and span mapping
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(reordered, f, ensure_ascii=False, indent=4)

    with open(f"{split_name}_{target_lang}_mapping.json", "w", encoding="utf-8") as f:
        json.dump(span_mapping, f, ensure_ascii=False, indent=4)

    output_dir = Path(f"{DATA_DIR}/{target_lang}{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_tok_file(reordered, output_dir / f"{target_lang}{output_name}_train.tok")
    save_rels_file(split_name, span_mapping, reordered, output_dir / f"{target_lang}{output_name}_train.rels")

    # Move intermediate files
    Path("intermediate").mkdir(exist_ok=True)
    shutil.move(file_name, f"intermediate/{file_name}")
    shutil.move(f"{split_name}_{target_lang}_mapping.json", f"intermediate/{split_name}_{target_lang}_mapping.json")

def main():
    split_name = "eng.rst.rstdt"
    split_name = f"{DATA_DIR}/{split_name}/{split_name}_train"
    output_name = ".rst.rstdt"
    lang = "ces"
    # lang = "deu"
    sentence_ids = get_original_text(split_name)
    translate_original_text_into_target_language(lang, sentence_ids, split_name)
    transform_to_dataset(lang, split_name, output_name)

if __name__ == "__main__":
    main()