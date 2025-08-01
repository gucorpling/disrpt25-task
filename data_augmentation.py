import disrptdata
from util import get_logger
from openai import OpenAI
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import time, io, json, re, stanza, shutil, random

MODEL = "gpt-4.1"
DATA_DIR = "data"
LANGUAGES = {"ces": "Czech", "deu": "German", "eus": "Basque", "fra": "French", "nld": "Dutch", "fas": "Persian"}
LANGUAGES_STANZA = {"ces": "cs", "deu": "de", "eus": "eu", "fra": "fr", "nld": "nl", "fas": "fa"}

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

def merge_spans(a, b):

    result = set()

    a = [tuple(map(int, span.split('-'))) if '-' in span else (int(span), int(span)) for span in a]
    b = [tuple(map(int, span.split('-'))) if '-' in span else (int(span), int(span)) for span in b]

    a_sorted = sorted(a)
    b_sorted = sorted(b)

    for start_a, end_a in a_sorted:
        for start_b, end_b in b_sorted:
            if start_a <= end_b and end_a >= start_b:
                result.add(f"{max(start_a, start_b)}-{min(end_a, end_b)}")
            elif end_a < start_b:
                break
    return result

def get_original_text(dir_name):    
    # rels
    data = io.open((dir_name + ".rels"), encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    split_lines = [line.split("\t") for line in lines[1:]]

    rels_ranges = defaultdict(set)

    for line in split_lines:
        doc = line[0]
    
        for i in range(1, 3):
            rels_ranges[doc].update(line[i].split(','))

    # conllu
    conllu_ranges = defaultdict(set) 
    doc_id = None
    doc_token_offset = 0 

    with open(dir_name + ".conllu", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    current_sentence = []

    for line in lines:
        if line.startswith("#"):
            if doc_id and current_sentence:
                conllu_ranges[doc_id].add(f"{doc_token_offset + 1}-{doc_token_offset + len(current_sentence)}")
            if line.startswith("# newdoc id ="):
                doc_id = line.split("=", 1)[1].strip()
                doc_token_offset = 0
            current_sentence = []
        elif line.startswith("# sent_id ="):
            continue
        elif line.strip() == "":
            if current_sentence:
                conllu_ranges[doc_id].add(f"{doc_token_offset + 1}-{doc_token_offset + len(current_sentence)}")
                doc_token_offset += len(current_sentence)
            current_sentence = []
        else:
            parts = line.split("\t")
            token_id = parts[0]
            if token_id.isdigit() and "-" not in token_id and "." not in token_id:
                current_sentence.append(line)
            else:
                pass

    # merge
    sentence_ids = defaultdict(set)
    for key in rels_ranges:
        sentence_ids[key] = merge_spans(rels_ranges[key], conllu_ranges[key])

    return sentence_ids
def get_label_distribution(dir_name):
    distribution = defaultdict(int)

    data = io.open((dir_name + ".rels"), encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    split_lines = [line.split("\t") for line in lines[1:]]

    for line in split_lines:
        label = line[-1]
        distribution[label] += 1
        
    return distribution

def get_filtered_data(split_name, lang, dataset_name):
    output_path = f"{split_name}_{lang}.rels"
    label_distribution = get_label_distribution(dataset_name)
    # print(label_distribution)

    filtered_lines = []
    classify_lines_by_labels = defaultdict(list)

    data = io.open((split_name + ".rels"), encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]

    # rstdt
    # for line in split_lines:
        # if line[0] == "wsj_1138":
            # break
        # classify_lines_by_labels[line[-1]].append("\t".join(line))
    # GUM
    for line in split_lines:
        if ("GUM_bio" in line[0] or "GUM_news" in line[0] or "GUM_letter" in line[0]) and "," not in line[1] and "," not in line[2]:
            classify_lines_by_labels[line[-1]].append("\t".join(line))
    # others
    for line in split_lines:
        classify_lines_by_labels[line[-1]].append("\t".join(line))

    for key, value in label_distribution.items():
        print(f"{key}: {len(classify_lines_by_labels[key])}_{value*0.75}")

    for label in classify_lines_by_labels:
       random.shuffle(classify_lines_by_labels[label])
       num = round(label_distribution.get(label, 0) * 0.76)
    #    num = label_distribution.get(label, 0)
       filtered_lines.extend(classify_lines_by_labels[label][0: num])

    filtered_lines = [header] + filtered_lines
    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_lines))

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

    data = disrptdata.get_segs_and_toks_for_docs_from_conllu(split_name)[2]

    translations = defaultdict(dict)

    for doc in sentence_ids:
        doc_content = data[doc]
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
    """Remap original span to new span using mapping dictionary, handling split spans."""
    result = []

    for part in input_str.split(","):
        if "-" not in part:
            part = f"{part}-{part}"
        if part in mapping:
            mapped = mapping[part]
            if "-" in mapped:
                start, end = mapped.split("-")
                result.append(start if start == end else mapped)
            else:
                result.append(mapped)
        else:
            start, end = map(int, part.split("-"))
            found = False
            mapped_start = 10**100
            mapped_end= 0

            for sub_span in mapping:
                if "-" in sub_span:
                    sub_start, sub_end = map(int, sub_span.split('-'))
                else:
                    sub_start = int(sub_span)
                    sub_end = int(sub_span)
                if sub_start >= start and sub_end <= end:
                    mapped = mapping[sub_span]
                    if "-" in mapped:
                        s, e = map(int, mapping[sub_span].split('-'))
                    else:
                        s = int(mapped)
                        e = int(mapped)
                    
                    if s < mapped_start:
                        mapped_start = s
                    if e > mapped_end:
                        mapped_end = e
                    found = True
            if mapped_start != mapped_end:
                result.append(f"{mapped_start}-{mapped_end}")
            else:
                result.append(str(mapped_start))

            if not found:
                raise ValueError(f"Span '{part}' not found in mapping or is not split correctly.")
    
    return ",".join(result)

def get_translate_spans(span_str, token_dict):
    """Join tokens from token_dict for each span, and combine multiple spans with <*>."""
    parts = []

    for span in span_str.split(","):
        if "-" not in span:
            span = f"{span}-{span}"
        if span not in token_dict:
            found = False
            sentence = ""
            if "-" in span:
                start, end = map(int, span.split('-'))
            else:
                start = int(span)
                end = int(span)

            for sub_span in token_dict:
                if "-" in sub_span:
                    sub_start, sub_end = map(int, sub_span.split('-'))
                else:
                    sub_start = int(sub_span)
                    sub_end = int(sub_span)
                if sub_start >= start and sub_end <= end:
                    sentence += " ".join(token_dict[sub_span])
                    found = True
            parts.append(sentence)
            if not found:
                raise ValueError(f"Span '{span}' not found in token_dict or is not split correctly.")
        else:
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
            # print(data[doc][sentence_id])
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

def save_rels_file(split_name, lang, mapping, translate_data, output_path):
    data = io.open(f"{split_name}_{lang}.rels", encoding="utf-8").read().strip().replace("\r", "")
    lines = data.split("\n")
    header = lines[0]
    split_lines = [line.split("\t") for line in lines[1:]]

    for line in split_lines:
        doc = line[0]
        line[5] = get_translate_spans(line[1], translate_data[doc])
        line[6] = get_translate_spans(line[2], translate_data[doc])
        line[9] = get_translate_spans(line[7], translate_data[doc])
        line[10] = get_translate_spans(line[8], translate_data[doc])
        line[1] = remap_spans(line[1], mapping[doc])
        line[2] = remap_spans(line[2], mapping[doc])
        line[3] = line[4] = ""
        line[7] = remap_spans(line[7], mapping[doc])
        line[8] = remap_spans(line[8], mapping[doc])

    modified_lines = ["\t".join(map(str, line)) for line in split_lines]
    final_output = [header] + modified_lines

    with io.open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output))

def parse_span(span_str):
    if "-" in span_str:
        start, end = map(int, span_str.split("-"))
    else:
        start = end = int(span_str)
    return start, end

def get_combined_translation(doc_id, start, end, span_mapping, reordered):
    tokens = []
    for orig_span, trans_span in sorted(span_mapping[doc_id].items(), key=lambda x: int(x[0].split("-")[0])):
        orig_start, orig_end = parse_span(orig_span)
        if orig_start > end:
            break
        if orig_end < start:
            continue
        if orig_start >= start and orig_end <= end:
            trans_tokens = reordered[doc_id][orig_span]
            tokens.extend(trans_tokens)
    return tokens

def save_conllu_file(conllu_path, reordered, span_mapping, output_path):
    with open(f"{conllu_path}.conllu", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    output_lines = []
    doc_id = None
    sent_id = None
    current_metadata = []
    current_sentence = []

    doc_token_offset = 0

    skip_doc = False

    for line in lines + [""]:
        if line.startswith("#"):
            current_metadata.append(line)
            if line.startswith("# newdoc id ="):
                doc_id = line.split("=", 1)[1].strip()
                if doc_id not in reordered or doc_id not in span_mapping:
                    print(f"⚠️ Skipping doc {doc_id}: not found in JSON data.")
                    skip_doc = True
                else:
                    skip_doc = False
                doc_token_offset = 0
            elif line.startswith("# sent_id ="):
                sent_id = line.split("=", 1)[1].strip()
        elif line.strip() == "":
            if skip_doc:
                current_metadata = []
                current_sentence = []
                continue


            token_lines = [
                line for line in current_sentence
                if line and line[0].isdigit() and "." not in line.split("\t")[0] and "-" not in line.split("\t")[0]
            ]

            if not token_lines:
                output_lines.extend(current_metadata)
                output_lines.append("")
                current_metadata = []
                current_sentence = []
                continue

            start_token_id = doc_token_offset + 1
            end_token_id = doc_token_offset + len(token_lines)
            original_range = f"{start_token_id}-{end_token_id}"

            translated_tokens = get_combined_translation(doc_id, start_token_id, end_token_id, span_mapping, reordered)

            if not translated_tokens:
                print(f"⚠️ Warning: No tokens found for {doc_id}:{start_token_id}-{end_token_id}")
                output_lines.extend(current_metadata)
                output_lines.extend(current_sentence)
            else:
                output_lines.extend(current_metadata)
                for i, token in enumerate(translated_tokens):
                    seg = "B-seg" if i == 0 else "O"
                    output_lines.append(f"{i+1}\t{token}\t_\t_\t_\t_\t_\t_\t_\tSeg={seg}")

            output_lines.append("")
            doc_token_offset += len(token_lines)

            current_metadata = []
            current_sentence = []

        else:
            current_sentence.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))



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
    save_rels_file(split_name, target_lang, span_mapping, reordered, output_dir / f"{target_lang}{output_name}_train.rels")
    save_conllu_file(split_name, reordered, span_mapping, output_dir / f"{target_lang}{output_name}_train.conllu")

    # Move intermediate files
    target_path = Path("intermediate") / file_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(file_name, f"intermediate/{file_name}")
    shutil.move(f"{split_name}_{target_lang}_mapping.json", f"intermediate/{split_name}_{target_lang}_mapping.json")
    shutil.move(f"{split_name}_{target_lang}.rels", f"intermediate/{split_name}_{target_lang}.rels")

def main():
    split_name = "eng.pdtb.gum"
    split_name = f"{DATA_DIR}/{split_name}/{split_name}_train"
    output_name = ".pdtb.gum"
    lang = "deu"
    # lang_dataset_name = "fas.rst.prstc"
    # lang_dataset_name = f"{DATA_DIR}/{lang_dataset_name}/{lang_dataset_name}_train"
    # sentence_ids = get_original_text(split_name)
    # translate_original_text_into_target_language(lang, sentence_ids, split_name)

    # get_filtered_data(split_name, lang, lang_dataset_name)
    transform_to_dataset(lang, split_name, output_name)

if __name__ == "__main__":
    main()