# Installation

## Requirements

- Python 3.10 or 3.11
- PyTorch >= 2.0 (CUDA 11.8+ recommended for GPU training; MPS supported on Apple Silicon)
- ~2 GB disk space for the environment

## Setup

**1. Create a conda environment**

```bash
conda create -n egogaze python=3.11
conda activate egogaze
```

**2. Install PyTorch**

Follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your platform and CUDA version. For example:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU / MPS (Apple Silicon)
pip install torch torchvision
```

**3. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

## Notes

- `torch.compile` (used in the full-run configs) requires PyTorch >= 2.0 and adds ~1 min compilation overhead on the first forward pass.
- Automatic Mixed Precision (`use_amp: true`) is enabled by default and requires a CUDA GPU or Apple MPS. Disable it by setting `use_amp: false` in your config for CPU-only runs.
- Weights & Biases logging is enabled by default. Disable with `wandb: {enabled: false}` in your config, or set the environment variable `WANDB_MODE=disabled`.
