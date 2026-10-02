"""
Pre-build the Ego-Exo-4D metadata pickle cache before training.

The datasets.py EgoExo4DDataset / EgoExo4DSequenceDataset scan all gaze CSVs
at init time and save a pickle cache. On the first run this can take a few
minutes (scanning thousands of CSVs). Running this script once creates the
cache so training starts instantly.

Cache files:
    {cache_dir}/egoexo4d_samples_train_b.pkl   ← Option B, train split
    {cache_dir}/egoexo4d_samples_val_b.pkl     ← Option B, val split
    (etc. for other splits / options)

Usage:
    python data/preprocessing_metadata_only.py \\
        --data_root   /path/to/EgoExo4D \\
        --splits_json configs/egoexo4d_splits.json \\
        --cache_dir   /path/to/EgoExo4D/.cache \\
        --options     a b \\
        --splits      train val

The script imports the internal helpers from src/datasets.py directly, so the
cache format is guaranteed to be identical to what training expects.
"""

import sys
import os
import argparse
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build Ego-Exo-4D metadata pickle cache"
    )
    parser.add_argument("--data_root",    required=True,
                        help="Ego-Exo-4D root directory")
    parser.add_argument("--splits_json",  required=True,
                        help="Path to egoexo4d_splits.json")
    parser.add_argument("--cache_dir",    required=True,
                        help="Output directory for .pkl cache files")
    parser.add_argument("--options",      nargs="+", default=["b"],
                        choices=["a", "b"],
                        help="Which option tier(s) to cache (default: b)")
    parser.add_argument("--splits",       nargs="+",
                        default=["train", "val"],
                        choices=["train", "val",
                                 "downstream_test", "generalization_test"],
                        help="Which split(s) to cache (default: train val)")
    args = parser.parse_args()

    # Import here so sys.path is already set up
    from src.datasets import _load_split_entries, _build_samples

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ego-Exo-4D Metadata Cache Builder")
    print("=" * 60)
    print(f"data_root  : {args.data_root}")
    print(f"splits_json: {args.splits_json}")
    print(f"cache_dir  : {args.cache_dir}")
    print(f"options    : {args.options}")
    print(f"splits     : {args.splits}")
    print("=" * 60)

    for option in args.options:
        for split in args.splits:
            cache_path = cache_dir / f"egoexo4d_samples_{split}_{option}.pkl"

            print(f"\n[option={option}, split={split}]")

            if cache_path.exists():
                import pickle
                with open(cache_path, "rb") as f:
                    existing = pickle.load(f)
                print(f"  Cache already exists: {len(existing):,} samples  ({cache_path})")
                print(f"  Skipping. Use --force to rebuild (not yet implemented).")
                continue

            # Load split entries
            entries = _load_split_entries(args.splits_json, option, split)
            print(f"  Split entries: {len(entries)}")
            if not entries:
                print(f"  No entries found for option={option}, split={split}. Skipping.")
                continue

            # Build samples (scans all gaze CSVs for these takes)
            print(f"  Scanning gaze CSVs and building sample index …")
            df = _build_samples(args.data_root, entries, cache_path=cache_path)
            print(f"  ✓ Cached {len(df):,} samples → {cache_path}")

    print("\n" + "=" * 60)
    print("Cache build complete.")
    print("=" * 60)

    # Print summary of what was cached
    print("\nCache files in", args.cache_dir, ":")
    for p in sorted(cache_dir.glob("egoexo4d_samples_*.pkl")):
        import pickle
        with open(p, "rb") as f:
            df = pickle.load(f)
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<45}  {len(df):>8,} samples   {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
