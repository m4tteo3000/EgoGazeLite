# Getting Started

This guide walks through the full pipeline from raw data to trained model and evaluation results.

## Step 1: Prepare the data

**Download Ego-Exo4D.** Request access and download via the [official CLI](https://ego-exo4d-data.org/). You need the `takes/` directory with Aria RGB videos and eye gaze CSVs.

**Generate splits (optional).** The pre-generated splits are already included in `configs/egoexo4d_splits.json`. Regenerating them from scratch (e.g. with a different seed or domain selection) additionally requires a `rekimoto_selection.json` listing the take UIDs of the downstream test set:

```bash
python data/generate_split.py \
    --takes_json     /path/to/egoexo4d/takes.json \
    --rekimoto_json  /path/to/rekimoto_selection.json \
    --output         configs/egoexo4d_splits.json
```

**Build the metadata cache.** This scans all gaze CSVs once and saves a pickle cache so training starts instantly on subsequent runs:

```bash
python data/preprocessing_metadata_only.py \
    --data_root   /path/to/EgoExo4D \
    --splits_json configs/egoexo4d_splits.json \
    --cache_dir   /path/to/EgoExo4D/.cache
```


## Step 2: Configure

Open `configs/full_run/full_run_option_a_cotrain.yaml` (Option A, the setup used in the paper) and update the paths for your machine:

```yaml
data_root:   "/path/to/EgoExo4D"
splits_json: "configs/egoexo4d_splits.json"
cache_dir:   "/path/to/EgoExo4D/.cache"
frames_dir:  null   # or a folder of pre-extracted frames (see comment in the config)
```


## Step 3: Train

Training runs all three stages sequentially:

- **Stage 1** — Saliency decoder (EfficientNet-Lite4 backbone + temporal difference + decoder)
- **Stage 2** — Attention transition (I-DT fixation detection + two-layer LSTM)
- **Stage 3** — Residual fusion of both pathways

```bash
python train.py --config configs/full_run/full_run_option_a_cotrain.yaml
```

Checkpoints are saved to the config's `checkpoint_dir` (`experiments/full_run_egoexo_option_a_cotrain_v1/`). To resume from a checkpoint:

```bash
python train.py --config configs/full_run/full_run_option_a_cotrain.yaml \
    --resume experiments/full_run_egoexo_option_a_cotrain_v1/checkpoint_stage2_best.pt \
    --stage 3
```

Training logs to Weights & Biases by default. Set `wandb: {enabled: false}` in the config to disable.

## Step 4: Evaluate

```bash
python evaluate.py \
    --checkpoint experiments/full_run_egoexo_option_a_cotrain_v1/checkpoint_final.pt \
    --data_root  /path/to/EgoExo4D \
    --cache_dir  /path/to/EgoExo4D/.cache \
    --option     a
```

For held-out evaluation on the generalization split (Basketball), add `--split generalization_test`.

## Expected results

In-distribution results on the Ego-Exo4D validation set for the Option A checkpoint (from the paper):

| Domain | AUC | AAE (°) | Pixel Dist. |
|--------|-----|---------|-------------|
| Cooking | 0.9705 | 7.58 | 24.46 |
| Soccer | 0.9702 | 7.67 | 23.68 |
| Health | 0.9684 | 7.73 | 25.51 |
| Bike Repair | 0.9668 | 8.03 | 25.76 |
| Dance | 0.9587 | 9.34 | 28.97 |
| Bouldering | 0.9566 | 9.64 | 30.16 |
| Music | 0.9545 | 10.23 | 30.76 |
| **Overall** | **0.9655** | **8.33** | **26.35** |

Model capacity compared with prior dual-process and transformer gaze models:

| Model | Params | GFLOPs |
|-------|--------|--------|
| EgoGazeLite | **15.7M** | **6.71** |
| Huang et al. (2018) | 51.0M | 57.76 |
| Lai et al. (2023) | 70.2M | 94.22 |

On an iPhone 15 Pro, the full gaze-and-crop pipeline takes 21.6 ms per frame end-to-end (≈46 FPS, P95 30.8 ms), of which the EgoGazeLite forward pass on the Neural Engine takes 8.9 ms.

*Downstream MLLM description-quality results are in the [paper](https://arxiv.org/abs/2608.15614).*
