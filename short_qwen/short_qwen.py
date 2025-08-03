import numpy as np
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import from transformers

from metrics import *  # Assuming metrics is a custom module

def sample_top_p(probs: torch.Tensor, p: float):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        return_hiddens: Optional[bool] = False
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.
            (Optional) return_hiddens (bool): Whether to return hidden states. Defaults to False.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
            (Optional) List[torch.Tensor]: Hidden states for each transformer block.
        """
        outputs = self.model(input_ids=tokens, output_hidden_states=return_hiddens)
        logits = outputs.logits
        hiddens = outputs.hidden_states if return_hiddens else None
        return logits, hiddens if return_hiddens else logits

class ShortQwen:
    def __init__(self, model_name: str, n_prune_layers: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # self.model = TransformerWrapper(self.model)  # wrap transformer to collect hidden states
        self.n_prune_layers = n_prune_layers

        # Attempt to access the correct attribute depending on model structure
        # Adjust the layers based on the model's internal architecture
        # print(dir(self.model)) 
        # print(dir(self.model.model)) 
        try:
            self.importances = [0 for _ in range(self.model.config.num_hidden_layers)]
            # self.importances = [0 for _ in range(len(self.model.model.h))]  # llama layer-wise importance scores
        except AttributeError:
            print("Model does not have 'h / layers' attribute, please verify the model structure.")

    def remove_layers(
        self,
        layers_to_remove: Optional[list[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                # kexin debug
                del self.model.model.layers[layer_idx] #transformer
                # del self.model.model.h[layer_idx] #llama
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool):
        n = 1
        if angular:
            assert self.n_prune_layers is not None, "Set number of layers to prune to use angular importance"
            n = self.n_prune_layers

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            if angular:
                # use only last token for angular distance as described in section 3.2
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: Optional[int] = 0,
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.9,
        angular: Optional[bool] = False
    ):
        """
        Computes layer-wise importances over input tokens.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            (Optional) max_gen_len (int): Maximum length of the generated text sequence.
            (Optional) temperature (float): Temperature value for controlling randomness in sampling.
            (Optional) top_p (float): Top-p probability threshold for nucleus sampling.
            (Optional) angular (bool): Whether to use angular distance.

        Returns:
            None
        """
        bsz = len(prompt_tokens)
        # assert bsz <= self.model.model.config.max_batch_size, (bsz, self.model.model.config.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.model.model.config.max_position_embeddings
        total_len = min(self.model.model.config.max_position_embeddings, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        
        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) &amp; (
                next_token == self.tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        # Compute block influence over full sequences
        outputs = self.model(tokens, output_hidden_states=True)
        hiddens = outputs.hidden_states #tuple
        #for i, hidden in enumerate(hiddens):
        #    print(f"Layer {i} hidden state shape: {hidden.shape}")
        # _, hiddens = self.model.forward(tokens, 0, return_hiddens=True)
        self.compute_bi(hiddens, angular=angular)
        return

if __name__ == "__main__":
    pass