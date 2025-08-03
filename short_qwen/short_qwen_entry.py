from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from short_qwen import ShortQwen  # ShortQwen class for model pruning
import os
import json
import shutil
import argparse
import warnings
warnings.simplefilter('ignore')

def prune(model_name, prune_layers):

    data = load_dataset("openai/MMMLU", "default", split="test").take(500) 
    def concatenate_fields(batch):
        return f"Question: {batch['Question'][0]} A: {batch['A'][0]} B: {batch['B'][0]} C: {batch['C'][0]} D: {batch['D'][0]} Answer: {batch['Answer'][0]}"
    data_split = data.map(lambda x: {"text": concatenate_fields(x)})

    dataloader = DataLoader(
        data_split,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator(device="cpu")
    )

    MAX_SEQ_LEN = 1024  # Set context width to 1024 for Qwen

    qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen_model = AutoModelForCausalLM.from_pretrained(model_name)

    short_qwen = ShortQwen(model_name=model_name, n_prune_layers=prune_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short_qwen.model.model.to(device)
    print(short_qwen.model.model) #qwen
    short_qwen.model.generate(
        input_ids=qwen_tokenizer("I am an avid fan of ", return_tensors="pt").input_ids.cuda(),
        max_length=20,
        use_cache=False
    )
    # Run the evaluation loop for pruning importance
    for batch in tqdm(dataloader, desc="Processing batches"):
        prompts = batch['text']

        # Tokenize the prompts / llama:bos_token=True,eos_token=False Qwen:
        prompt_tokens = [qwen_tokenizer.encode(x, return_tensors="pt").squeeze(0).cuda() for x in prompts]
        max_prompt_len = max(len(t) for t in prompt_tokens)

        # Sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, 256):
            inputs = [p[start:start+MAX_SEQ_LEN] for p in prompt_tokens if len(p) > start]

            short_qwen.eval_importance(
                prompt_tokens=inputs,
                max_gen_len=0
            )

    # Print the layer importance scores and remove layers accordingly
    print("[^V^] importances: ", short_qwen.importances)
    print("[O.O] remove layers: ", short_qwen.remove_layers())

    # Check the model layers after pruning
    print(short_qwen.model.model.layers)
    print(f"Model layers after pruning: {len(short_qwen.model.model.layers)}")

    # ================================================ save pruned Qwen ========================================================
    def save_pruned_qwen_model(short_qwen, model_name, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(short_qwen.model.state_dict(), f'{save_directory}/pytorch_model.bin')

        original_model_dir = model_name
        
        config_path = os.path.join(original_model_dir, 'config.json')
        generation_config_path = os.path.join(original_model_dir, 'generation_config.json')
        merges_path = os.path.join(original_model_dir, 'merges.txt')
        tokenizer_json_path = os.path.join(original_model_dir, 'tokenizer.json')
        tokenizer_config_path = os.path.join(original_model_dir, 'tokenizer_config.json')
        vocab_path = os.path.join(original_model_dir, 'vocab.json')
        
        print(config_path, save_directory)
        shutil.copy(config_path, os.path.join(save_directory, 'config.json'))
        shutil.copy(generation_config_path, os.path.join(save_directory, 'generation_config.json'))
        shutil.copy(merges_path, os.path.join(save_directory, 'merges.txt'))
        shutil.copy(tokenizer_json_path, os.path.join(save_directory, 'tokenizer.json'))
        shutil.copy(tokenizer_config_path, os.path.join(save_directory, 'tokenizer_config.json'))
        shutil.copy(vocab_path, os.path.join(save_directory, 'vocab.json'))

        with open(os.path.join(save_directory, 'config.json'), 'r') as f:
            config = json.load(f)
        config['num_hidden_layers'] = len(short_qwen.model.model.layers)
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_directory)
        print(f"The pruned model and tokenizer have been saved to: {save_directory}")

    # 🌟 Step4: Save the pruned Qwen model
    print("[^V^]Save the pruned Qwen model!")
    # "-"+str(len(qwen_model.model.model.layers))+"to"+str(len(short_qwen.model.model.layers))+"-"
    save_directory="output/"+model_name.split("/")[-1]+"-pruned"+"-"+str(qwen_model.config.num_hidden_layers)+"to"+str(len(short_qwen.model.model.layers))+"-"+str(prune_layers)+"prune_layers"
    save_pruned_qwen_model(short_qwen, model_name, save_directory)

    # ================================================ load pruned Qwen &amp; generate ========================================================
    def load_pruned_qwen_model(model_name, save_directory):
        qwen_model = AutoModelForCausalLM.from_pretrained(save_directory)
        print("Config layers:", qwen_model.config.num_hidden_layers)
        qwen_model.config.num_hidden_layers = len(short_qwen.model.model.layers)  
        print("Config layers:", qwen_model.config.num_hidden_layers)
        qwen_model.config.use_cache = False
        print(f"The pruned model has been loaded from {save_directory}")
        return qwen_model

    # Load the pruned Qwen model
    print("[^V^]loading the pruned Qwen model!")
    pruned_qwen = load_pruned_qwen_model(model_name, save_directory)

    # 🌟 Step5: Sample text completion after pruning
    generated = short_qwen.model.generate(
        # input_ids=qwen_tokenizer("I am an avid fan of ", return_tensors="pt").input_ids.cuda(),
        input_ids=qwen_tokenizer("请你翻译成英文: 香港代购SK2神仙水限量版", return_tensors="pt").input_ids.cuda(), 
        max_length=20,
        use_cache=False  
    )
    print("Generated text:", qwen_tokenizer.decode(generated[0], skip_special_tokens=True))

if __name__ == "__main__":
    model_name = "output/Qwen3-4B-Base"
    prune_layers = 1
    prune(model_name, prune_layers)

    