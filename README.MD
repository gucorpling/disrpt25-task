# Georgetown University Team for DISRPT 2025 Shared Task

## Setup

We use conda for maintaining a python Dev environment and `requirements.txt` for cataloguing the dependencies
1. Create (or activate) the environment.
```bash
conda create -n disrpt python==3.9
conda activate disrpt
```

2. Install dependencies
Dependencies can be installed with
```
pip install -r requirements.txt
```

## Preparing Data
Clone the shared task public repo (or add temporarily as submodule)
Change the CLI directory to the shared task repo `sharedtask2025/util` and run the process underscore.py
Once the process is complete copy the data to `/data`

The `.gitignore` is updated to not commit the data but it can change if we decide to change that


## Dependency Management
However, we only use pip for dependency management.
All the dependencies for the package must be added to the `requirements.txt`
