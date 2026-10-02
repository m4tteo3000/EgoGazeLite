"""
Generate official Ego-Exo4D train/val/test splits for gaze prediction.

Split design:
  - downstream_test     : Rekimoto 138 takes (held out for downstream evaluation)
  - generalization_test : Basketball domain (held out for cross-domain evaluation)
  - train / val         : everything else, domain-balanced subsampling

Two subsampling tiers encoded in the same file:
  option_a  ~4h per domain
  option_b  ~10h per domain

Usage:
    python data/generate_split.py \\
        --takes_json        /path/to/egoexo4d/takes.json \\
        --rekimoto_json     /path/to/rekimoto_selection.json \\
        --output            configs/egoexo4d_splits.json

Output:
    egoexo4d_splits.json  — referenced by train.py and data/preprocessing_metadata_only.py
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
VAL_RATIO   = 0.20

GENERALIZATION_DOMAIN = "Basketball"

TRAINING_DOMAINS = [
    "Cooking",
    "Rock Climbing",
    "Dance",
    "Health",
    "Music",
    "Bike Repair",
    "Soccer",
]

OPTION_A_HOURS = 4.0
OPTION_B_HOURS = 10.0
# ──────────────────────────────────────────────────────────────────────────────


def load_takes(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("takes", list(data.values())[0])


def load_rekimoto_uids(path: Path) -> set:
    with open(path) as f:
        sel = json.load(f)
    return {
        t["take_uid"]
        for takes in sel["categories"].values()
        for t in takes
    }


def sample_by_hours(takes: list, target_hours: float, seed: int) -> list:
    rng = random.Random(seed)
    shuffled = takes[:]
    rng.shuffle(shuffled)

    selected, accumulated = [], 0.0
    for t in shuffled:
        dur = t.get("duration_sec", 0) / 3600
        if accumulated + dur <= target_hours:
            selected.append(t)
            accumulated += dur
        if accumulated >= target_hours:
            break
    return selected


def train_val_split(takes: list, val_ratio: float, seed: int) -> tuple:
    """Participant-stratified train/val split — no participant in both sets."""
    by_participant = defaultdict(list)
    for t in takes:
        pid = t.get("participant_uid", "unknown")
        by_participant[pid].append(t)

    rng = random.Random(seed)
    train, val = [], []
    for pid, ptakes in by_participant.items():
        shuffled = ptakes[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def make_entry(take: dict, split: str, option_a: bool, option_b: bool) -> dict:
    return {
        "take_uid":        take["take_uid"],
        "take_name":       take["take_name"],
        "domain":          take.get("parent_task_name", ""),
        "task_name":       take.get("task_name", ""),
        "participant_uid": take.get("participant_uid"),
        "duration_sec":    round(take.get("duration_sec", 0), 2),
        "split":           split,
        "option_a":        option_a,
        "option_b":        option_b,
    }


def cnt(entries, split, option=None):
    f = [e for e in entries if e["split"] == split]
    if option == "a": f = [e for e in f if e["option_a"]]
    if option == "b": f = [e for e in f if e["option_b"]]
    return len(f)


def hrs(entries, split, option=None):
    f = [e for e in entries if e["split"] == split]
    if option == "a": f = [e for e in f if e["option_a"]]
    if option == "b": f = [e for e in f if e["option_b"]]
    return sum(e["duration_sec"] for e in f) / 3600


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ego-Exo4D train/val/test splits for gaze prediction"
    )
    parser.add_argument("--takes_json", required=True,
                        help="Path to Ego-Exo4D takes.json")
    parser.add_argument("--rekimoto_json", required=True,
                        help="Path to rekimoto_selection.json (downstream test set definition)")
    parser.add_argument("--output", default="configs/egoexo4d_splits.json",
                        help="Output path for the splits JSON (default: configs/egoexo4d_splits.json)")
    args = parser.parse_args()

    takes_json = Path(args.takes_json)
    rekimoto_json = Path(args.rekimoto_json)
    output_json = Path(args.output)

    print("Loading takes.json...")
    takes = load_takes(takes_json)
    print(f"  {len(takes)} takes")

    print("Loading Rekimoto selection...")
    rekimoto_uids = load_rekimoto_uids(rekimoto_json)
    print(f"  {len(rekimoto_uids)} downstream test takes")

    # ── Bucket all takes ───────────────────────────────────────────────────────
    rekimoto_takes       = []
    generalization_takes = []
    training_pool        = defaultdict(list)
    skipped              = []

    for t in takes:
        if not t.get("has_trimmed_eye_gaze") or t.get("is_dropped"):
            continue
        uid    = t["take_uid"]
        domain = t.get("parent_task_name", "")

        if uid in rekimoto_uids:
            rekimoto_takes.append(t)
        elif domain == GENERALIZATION_DOMAIN:
            generalization_takes.append(t)
        elif domain in TRAINING_DOMAINS:
            training_pool[domain].append(t)
        else:
            skipped.append(t)

    print(f"\nAllocation:")
    print(f"  downstream_test:                      {len(rekimoto_takes)} takes")
    print(f"  generalization_test ({GENERALIZATION_DOMAIN}): {len(generalization_takes)} takes")
    for d in sorted(training_pool):
        ts = training_pool[d]
        h  = sum(t["duration_sec"] for t in ts) / 3600
        print(f"  training pool — {d:<15}: {len(ts):>4} takes  ({h:.1f}h)")
    print(f"  skipped:                              {len(skipped)}")

    # ── Build entries ──────────────────────────────────────────────────────────
    entries = []

    for t in rekimoto_takes:
        entries.append(make_entry(t, "downstream_test", True, True))
    for t in generalization_takes:
        entries.append(make_entry(t, "generalization_test", True, True))

    print(f"\nSampling training pool:")
    for domain in sorted(training_pool):
        domain_takes = training_pool[domain]
        total_h      = sum(t["duration_sec"] for t in domain_takes) / 3600

        sample_a = sample_by_hours(domain_takes, OPTION_A_HOURS, RANDOM_SEED)
        sample_b = sample_by_hours(domain_takes, OPTION_B_HOURS, RANDOM_SEED)

        h_a = sum(t["duration_sec"] for t in sample_a) / 3600
        h_b = sum(t["duration_sec"] for t in sample_b) / 3600
        print(f"  {domain:<15}: A={len(sample_a)} ({h_a:.1f}h)  "
              f"B={len(sample_b)} ({h_b:.1f}h)  [avail: {total_h:.1f}h]")

        uids_a = {t["take_uid"] for t in sample_a}
        uids_b = {t["take_uid"] for t in sample_b}

        _, val_a   = train_val_split(sample_a, VAL_RATIO, RANDOM_SEED)
        _, val_b   = train_val_split(sample_b, VAL_RATIO, RANDOM_SEED)
        val_uids_a = {t["take_uid"] for t in val_a}
        val_uids_b = {t["take_uid"] for t in val_b}

        for t in domain_takes:
            uid  = t["take_uid"]
            in_a = uid in uids_a
            in_b = uid in uids_b

            if not in_a and not in_b:
                entries.append(make_entry(t, "unused", False, False))
                continue

            if in_b:
                split_label = "val" if uid in val_uids_b else "train"
            else:
                split_label = "val" if uid in val_uids_a else "train"

            entries.append(make_entry(t, split_label, in_a, in_b))

    # ── Write output ───────────────────────────────────────────────────────────
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "description":               "Official Ego-Exo4D splits for gaze prediction",
            "seed":                      RANDOM_SEED,
            "val_ratio":                 VAL_RATIO,
            "generalization_domain":     GENERALIZATION_DOMAIN,
            "option_a_hours_per_domain": OPTION_A_HOURS,
            "option_b_hours_per_domain": OPTION_B_HOURS,
            "stats": {
                "downstream_test":     cnt(entries, "downstream_test"),
                "generalization_test": cnt(entries, "generalization_test"),
                "option_a": {
                    "train":       cnt(entries, "train", "a"),
                    "val":         cnt(entries, "val",   "a"),
                    "train_hours": round(hrs(entries, "train", "a"), 1),
                    "val_hours":   round(hrs(entries, "val",   "a"), 1),
                },
                "option_b": {
                    "train":       cnt(entries, "train", "b"),
                    "val":         cnt(entries, "val",   "b"),
                    "train_hours": round(hrs(entries, "train", "b"), 1),
                    "val_hours":   round(hrs(entries, "val",   "b"), 1),
                },
            },
        },
        "takes": entries,
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓  {output_json.resolve()}")
    print(f"\n  downstream_test:     {cnt(entries, 'downstream_test')} takes")
    print(f"  generalization_test: {cnt(entries, 'generalization_test')} takes")
    print(f"\n  Option A ({OPTION_A_HOURS}h/domain):")
    print(f"    train: {cnt(entries, 'train', 'a')} takes ({hrs(entries, 'train', 'a'):.1f}h)")
    print(f"    val:   {cnt(entries, 'val',   'a')} takes ({hrs(entries, 'val',   'a'):.1f}h)")
    print(f"\n  Option B ({OPTION_B_HOURS}h/domain):")
    print(f"    train: {cnt(entries, 'train', 'b')} takes ({hrs(entries, 'train', 'b'):.1f}h)")
    print(f"    val:   {cnt(entries, 'val',   'b')} takes ({hrs(entries, 'val',   'b'):.1f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
