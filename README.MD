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
1. Original data processing: Clone the shared task repo (or add as a submodule), navigate to `sharedtask2025/util`, and run python `process_underscore.py`. Then copy the output data folder into this repo as `data/`.
2. Augmented data preparation: Copy the existing `augmented_data/` directory into `data/` to consolidate all datasets.


## Run

This project supports multi-GPU training and single-GPU evaluation via the `--mode` flag.

### Training

To train the model using multiple GPUs (e.g., 4 GPUs):

```bash
torchrun --nproc_per_node=4 decoder_w_aug.py --mode train --checkpoint_path output/
```

* `--mode train`: Specifies training mode.
* `--nproc_per_node=4`: Number of GPUs to use.
* `--checkpoint_path`: Directory to save model checkpoints.

### Single-GPU Training

To train the model on a single GPU, you can run the script directly with `python`:

```bash
python decoder_w_aug.py --mode train --checkpoint_path output/
```

#### Training Configuration
Our experiments were conducted with an effective batch size of 64. This was achieved using 4 GPUs, a per_device_batch_size of 1, and gradient_accumulation_steps set to 16 `(4 GPUs * 1 batch/GPU * 16 steps = 64)`.

For single-GPU training, set `gradient_accumulation_steps` to **64** in order to match the effective batch size used in our experiments.

---

### Evaluation

To evaluate the model (usually on a single GPU):

```bash
python decoder_w_aug.py --mode eval --checkpoint_path output/checkpoint-3827 --res_path res/
```

* `--mode eval`: Specifies evaluation mode.
* `--checkpoint_path`: Directory where the checkpoint is stored, from which the model will be loaded. 
  * As training is configured to save only the final model, there will be a single checkpoint folder. This path should point directly to that folder (e.g., `output/checkpoint-3827`).
* `--res_path`: Directory to save the prediction results.

To evaluate directly our final model:

```bash
python decoder_w_aug.py --mode eval --checkpoint_path JuNymphea/Georgetown-qwen3-4B-finetuned-for-disrpt2025 --res_path res/
```

## Model Checkpoints
- [Base model (before fine-tuning)](https://huggingface.co/JuNymphea/Georgetown-qwen3-4B-pruned-for-disrpt2025)

- [Fine-tuned model (trained on task data)](https://huggingface.co/JuNymphea/Georgetown-qwen3-4B-finetuned-for-disrpt2025)

