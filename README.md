# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation

ForceVLA is based on the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA). Both training and inference are based on π₀.

## Requirements

### Hardware

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

### Operating System

The repo has been tested with **Ubuntu 22.04 LTS**. We do not currently support other operating systems.

### System Dependencies (Ubuntu 22.04)

Install the required system-level packages before proceeding with the Python setup:

```bash
sudo apt-get update
sudo apt-get install -y \
    git git-lfs \
    build-essential clang \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libegl1-mesa-dev libgles2-mesa-dev \
    linux-headers-generic
```

## Dataset

https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data

## Installation

There are two ways to install the project: using **UV** (recommended) or using **Conda + pip**.

### Option A: Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager that handles all dependencies automatically.

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository with submodules
git clone --recurse-submodules https://github.com/OTaku9418/ForceVLA_exp.git
cd ForceVLA_exp

# 3. Install Python and all dependencies
uv python install
uv sync --all-extras --dev

# 4. Install flaxformer
cd flaxformer/
pip install -e .
cd ..
```

### Option B: Using Conda + pip

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules https://github.com/OTaku9418/ForceVLA_exp.git
cd ForceVLA_exp

# 2. Create conda environment
conda create -n forcevla python=3.11 -y
conda activate forcevla

# 3. Upgrade pip and install CUDA toolkit
python -m pip install --upgrade pip setuptools wheel
conda install -c nvidia cuda-toolkit=12.8

# 4. Install LeRobot
cd lerobot/
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .
cd ..

# 5. Install OpenPI
pip install -e .

# 6. Install dlimp (for RLDS/DROID data support)
cd dlimp/
pip install -e .
cd ..

# 7. Install openpi-client
cd packages/openpi-client/
pip install -e .
cd ../..

# 8. Install flaxformer
cd flaxformer/
pip install -e .
cd ..
```

> **Note:** If `lerobot/` or `dlimp/` directories are empty, initialize the submodules:
> ```bash
> git submodule update --init --recursive
> ```
> If that doesn't work, clone them manually:
> ```bash
> git clone https://github.com/huggingface/lerobot.git lerobot/
> git clone https://github.com/kvablack/dlimp.git dlimp/
> ```

## Train Policy

```bash
export HF_LEROBOT_HOME="<your_data_path>"
python scripts/compute_norm_stats.py --config-name forcevla_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py forcevla_lora \
    --exp-name=my_experiment \
    --overwrite \
    --batch_size 32 \
    --save_interval 2000 \
    --keep_period 10000
```

## Docker Setup

For Docker-based setup, see [docs/docker.md](docs/docker.md).
