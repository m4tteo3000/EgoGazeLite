# Dataset Preparation

EgoGazeLite uses the [Ego-Exo4D](https://ego-exo4d-data.org/) dataset.

## Download

1. Request access at [ego-exo4d-data.org](https://ego-exo4d-data.org/)
2. Install the Ego-Exo4D CLI:
   ```bash
   pip install ego4d
   ```
3. Download the required components:
   ```bash
   egoexo -o /path/to/EgoExo4D --parts takes annotations
   ```
   You need the Aria RGB videos (`frame_aligned_videos/downscaled/448/aria01_214-1.mp4`) and eye gaze CSVs (`eye_gaze/`).

## Expected Directory Structure

```
EgoExo4D/
├── takes/
│   ├── <take_name>/
│   │   ├── frame_aligned_videos/
│   │   │   └── downscaled/
│   │   │       └── 448/
│   │   │           └── aria01_214-1.mp4
│   │   └── eye_gaze/
│   │       └── personalized_eye_gaze_2d.csv
│   └── ...
└── takes.json
```

## Splits

The pre-generated splits file (`configs/egoexo4d_splits.json`) defines four subsets:

| Split | Description |
|-------|-------------|
| `train` | Domain-balanced training set |
| `val` | Validation set (participant-stratified) |
| `generalization_test` | Basketball domain (held out for cross-domain evaluation) |
| `downstream_test` | 138 takes for downstream evaluation |

Two data tiers are available:
- **Option A** — ~4h per domain (~28h total)
- **Option B** — ~10h per domain (~68h total, default)

To regenerate splits from scratch, see `data/generate_split.py`.
