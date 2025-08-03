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
1. Original data processing: Clone the shared task repo (or add as a submodule), navigate to sharedtask2025/util, and run python process_underscore.py. Then copy the output data folder into this repo as data/.
2. Augmented data preparation: Copy the existing augmented_data/ directory into data/ to consolidate all datasets.


## Run

To launch multi-GPU training or inference (e.g., using 4 GPUs), run:

```bash
torchrun --nproc_per_node=4 decoder_w_aug.py --checkpoint_path output/ --res_path res/
```

--nproc_per_node=4: Specifies the number of processes (GPUs) to use.

--checkpoint_path: Directory to save or load model checkpoints.

--res_path: Directory where the prediction results will be saved.

## Model Checkpoints
- [Base model (before fine-tuning)](https://huggingface.co/JuNymphea/Georgetown-qwen3-4B-pruned-for-disrpt2025)

- [Fine-tuned model (trained on task data)](https://huggingface.co/JuNymphea/Georgetown-qwen3-4B-finetuned-for-disrpt2025)
