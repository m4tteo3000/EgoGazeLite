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

Open `configs/full_run/full_run_option_b_cotrain.yaml` and update the paths for your machine:

```yaml
data_root:   "/path/to/EgoExo4D"
splits_json: "configs/egoexo4d_splits.json"
cache_dir:   "/path/to/EgoExo4D/.cache"
frames_dir:  null   # or a folder of pre-extracted frames (see comment in the config)
```


## Step 3: Train

Training runs all three stages sequentially:

- **Stage 1** — Saliency decoder (EfficientNet-Lite4 backbone + temporal difference + decoder)
- **Stage 2** — Attention transition (velocity-based fixation detection + LSTM)
- **Stage 3** — Gated fusion (residual fusion of both pathways)

```bash
python train.py --config configs/full_run/full_run_option_b_cotrain.yaml
```

Checkpoints are saved to the config's `checkpoint_dir` (`experiments/full_run_egoexo_option_b_cotrain_v1/`). To resume from a checkpoint:

```bash
python train.py --config configs/full_run/full_run_option_b_cotrain.yaml \
    --resume experiments/full_run_egoexo_option_b_cotrain_v1/checkpoint_stage2_best.pt \
    --stage 3
```

Training logs to Weights & Biases by default. Set `wandb: {enabled: false}` in the config to disable.

## Step 4: Evaluate

```bash
python evaluate.py \
    --checkpoint experiments/full_run_egoexo_option_b_cotrain_v1/checkpoint_final.pt \
    --data_root  /path/to/EgoExo4D \
    --cache_dir  /path/to/EgoExo4D/.cache \
    --option     b
```

For held-out evaluation on the generalization split (Basketball), add `--split generalization_test`.

## Expected results

Results on GTEA Gaze+ (OP02 held-out subject, 30,344 frames):

| Model | AUC | AAE_CoM (°) | Pixel Dist. | Params | GFLOPs | MPS FPS | iPhone 15 Pro NE FPS |
|-------|-----|------------|-------------|--------|--------|---------|----------------------|
| EgoGazeLite | **0.9592** | 5.97 | **29.17** | **15.7M** | **6.71** | **30** | **216** |
| Huang et al. (2018) | 0.957 | **4.0** | — | 50.4M | 95.2 | 19 | 48 |
| Lai et al. (2023) | 0.9434 | 6.85 | 33.46 | 70.4M | 57.1 | 1.3 | — ¹ |

¹ Lai et al. (2023) could not be converted to CoreML (unsupported op: `upsample_trilinear3d`).

*Full results and ablations in the paper.*
