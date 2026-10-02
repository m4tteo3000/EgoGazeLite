"""
Dataset classes for Ego-Exo-4D gaze prediction.
Replaces the EGTEA Gaze+ dataset classes.

Supports two frame-loading modes (same pattern as EGTEA pipeline):
    1. Pre-extracted JPGs  (frames_dir set)  — fast, recommended for GPU training
    2. On-the-fly mp4 decoding (frames_dir=None) — slower, no preprocessing needed

Dataset structure (on disk):
    {data_root}/takes.json
    {data_root}/takes/{name}/
        frame_aligned_videos/downscaled/448/aria01_214-1.mp4   (448×448, 30fps)
        eye_gaze/personalized_eye_gaze_2d.csv  (preferred)
        eye_gaze/general_eye_gaze_2d.csv       (fallback if personalized absent)

Pre-extracted frames layout:
    {frames_dir}/{take_name}/frame_{idx:06d}.jpg

Split file (in pipeline repo):
    configs/egoexo4d_splits.json   — domain-balanced custom splits

Key constants:
    ARIA_NATIVE_SIZE = 1408   Aria RGB native resolution (gaze coord space)
    TARGET_SIZE = (300, 300)  Model input / heatmap resolution
    GAZE_SKIP = 3             video_frame = gaze_frame_num × 3   (30fps / 10fps)
    SCALE_FACTOR_X/Y          Aria native → TARGET_SIZE scaling

Temporal alignment:
    Gaze is at 10fps, video at 30fps.
    video_frame_idx = gaze_frame_num * 3
    frame_t           = video frame at gaze_frame_num * 3
    frame_t_minus_1   = video frame at gaze_frame_num * 3 - 1  (1 video frame back)

Fixation detection:
    Ego-Exo-4D has no explicit fixation/saccade labels.
    I-DT (Identification by Dispersion Threshold) is pre-computed for sequence
    datasets (Stage 2/3), matching GazeLite.compute_fixation_state(method='idt')
    exactly so that training labels and inference behaviour are consistent.

    Parameters (mirror src/models/gaze_lite.py):
        IDT_WINDOW               = 7 frames  (FIXATION_IDT_WINDOW)
        IDT_DISPERSION_THRESHOLD = 8.0 px    (FIXATION_DISPERSION_THRESHOLD)

    Algorithm per frame t:
        Look at past window gaze_x[t-7 : t], gaze_y[t-7 : t]  (exclude frame t itself,
        matching the gaze_history slice in train_stage3_epoch).
        dispersion = (max_x − min_x) + (max_y − min_y)
        fixation = 1  if dispersion < 8.0  (or window has < 2 points)
                   0  otherwise (saccade)
"""

from __future__ import annotations   # enables X | Y type hints on Python 3.9

import json
import pickle
import threading

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms

# ── Resolution constants ───────────────────────────────────────────────────────
ARIA_NATIVE_SIZE  = 1408          # Aria RGB native resolution (square)
VIDEO_STREAM_SIZE = 448           # Pre-downscaled video size (square)
TARGET_SIZE       = (300, 300)    # Model input size (width, height)

VIDEO_STREAM = "aria01_214-1.mp4"  # preferred, fall back to any aria*_214-1.mp4
GAZE_FPS  = 10.0
VIDEO_FPS = 30.0
GAZE_SKIP = int(VIDEO_FPS / GAZE_FPS)   # = 3

# ── Gaze coordinate scaling ────────────────────────────────────────────────────
# Gaze coords in CSV are in native Aria space (~1408×1408).
# Scale to TARGET_SIZE (300×300) for model / loss computation.
SCALE_FACTOR_X = TARGET_SIZE[0] / ARIA_NATIVE_SIZE    # 300/1408 ≈ 0.2131
SCALE_FACTOR_Y = TARGET_SIZE[1] / ARIA_NATIVE_SIZE    # same (square sensor)

# Gaussian sigma: maintain same angular spread as EGTEA (70px @ 1280px ≈ 3.1°)
# 70 * (300 / 1408) ≈ 14.9px at 300px
GAUSSIAN_SIGMA = 70.0 * SCALE_FACTOR_X   # ≈ 14.9px

# ── I-DT fixation detection thresholds (mirror src/models/gaze_lite.py) ───────
# Must stay in sync with FIXATION_IDT_WINDOW and FIXATION_DISPERSION_THRESHOLD
# defined in GazeLite so that pre-computed labels match inference behaviour.
#
# Calibration note (Ego-Exo-4D vs EGTEA):
#   EGTEA runs at 24fps → 7 frames = 292ms window
#   Aria gaze runs at 10fps → 7 frames = 700ms (3× too long, kills most fixations)
#   Time-equivalent window at 10fps: round(7 * 10/24) = 3 frames = 300ms
#
#   Threshold: EGTEA 8px ≈ 1.6° at 60° FOV. Aria is a wearable tracker with
#   ~1–2° noise floor during physical tasks → threshold raised to 16px ≈ 4.2°
#   to avoid classifying tracker noise as saccades.
IDT_WINDOW               = 3      # frames to look back — 300ms at 10fps (≈ EGTEA's 7@24fps)
IDT_DISPERSION_THRESHOLD = 16.0   # px in TARGET_SIZE space — raised for Aria wearable noise


# ── Heatmap generation ─────────────────────────────────────────────────────────

def create_gaze_heatmap(
    gaze_x_native: float,
    gaze_y_native: float,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Create a normalised Gaussian gaze heatmap at TARGET_SIZE resolution.

    Args:
        gaze_x_native: Gaze x-coord in native Aria space (0–1408)
        gaze_y_native: Gaze y-coord in native Aria space (0–1408)
        sigma:         Gaussian spread in TARGET_SIZE pixels

    Returns:
        heatmap: float32 array [H, W] normalised to [0, 1]
    """
    tgt_w, tgt_h = TARGET_SIZE

    gx = int(round(gaze_x_native * SCALE_FACTOR_X))
    gy = int(round(gaze_y_native * SCALE_FACTOR_Y))
    gx = np.clip(gx, 0, tgt_w - 1)
    gy = np.clip(gy, 0, tgt_h - 1)

    heatmap = np.zeros((tgt_h, tgt_w), dtype=np.float32)
    heatmap[gy, gx] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# ── Video cache (thread-local, one per DataLoader worker) ─────────────────────

class VideoCache:
    """
    Thread-local frame loader with two modes:

    Fast path  (frames_dir set): reads pre-extracted 300×300 JPGs from disk.
        Path: {frames_dir}/{take_name}/frame_{frame_idx:06d}.jpg

    Slow path (frames_dir=None): decodes frames on-the-fly from .mp4 using
        a thread-local OpenCV VideoCapture cache.

    Each DataLoader worker gets its own independent state.
    """

    def __init__(self, max_handles: int = 10, frames_dir: str | None = None):
        self.max_handles = max_handles
        self.frames_dir  = Path(frames_dir) if frames_dir else None
        self._local      = threading.local()

    def _get_cache(self) -> tuple:
        if not hasattr(self._local, 'cache'):
            self._local.cache = {}
            self._local.order = []
        return self._local.cache, self._local.order

    def get_frame(self, video_path: str, frame_idx: int,
                  take_name: str | None = None) -> np.ndarray:
        """
        Return a single BGR frame at TARGET_SIZE resolution.

        Args:
            video_path: Absolute path to the .mp4 file (used in slow path)
            frame_idx:  0-indexed frame number
            take_name:  Required for the fast path JPG lookup
        """
        # ── Fast path: pre-extracted JPG ──────────────────────────────────
        if self.frames_dir is not None:
            if take_name is None:
                raise ValueError("take_name required when frames_dir is set")
            jpg_path = self.frames_dir / take_name / f"frame_{frame_idx:06d}.jpg"
            frame = cv2.imread(str(jpg_path))
            if frame is None:
                raise FileNotFoundError(f"Pre-extracted frame not found: {jpg_path}")
            # Already 300×300 — resize only if needed
            if frame.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
                frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            return frame

        # ── Slow path: on-the-fly mp4 decoding ───────────────────────────
        cache, order = self._get_cache()

        if video_path not in cache:
            if len(cache) >= self.max_handles:
                oldest = order.pop(0)
                cache[oldest].release()
                del cache[oldest]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {video_path}")
            cache[video_path] = cap
            order.append(video_path)
        else:
            order.remove(video_path)
            order.append(video_path)

        cap = cache[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        return cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    def release_all(self):
        cache, _ = self._get_cache()
        for cap in cache.values():
            cap.release()
        cache.clear()


# ── Split loading ──────────────────────────────────────────────────────────────

def _load_split_entries(splits_json: Path, option: str, split: str) -> list:
    """
    Return entries from egoexo4d_splits.json for the given option and split.

    Args:
        splits_json: Path to configs/egoexo4d_splits.json
        option:      'a' or 'b'
        split:       'train' or 'val'
    """
    _valid_splits = ('train', 'val', 'downstream_test', 'generalization_test')
    assert option in ('a', 'b'), f"option must be 'a' or 'b', got '{option}'"
    assert split in _valid_splits, \
        f"split must be one of {_valid_splits}, got '{split}'"

    flag_key = f'option_{option}'
    with open(splits_json) as f:
        data = json.load(f)

    # For downstream_test / generalization_test: option flag is not required —
    # these splits are defined regardless of option tier.
    if split in ('downstream_test', 'generalization_test'):
        return [e for e in data['takes'] if e.get('split') == split]

    return [e for e in data['takes'] if e.get(flag_key) and e.get('split') == split]


# ── Metadata building (cached) ─────────────────────────────────────────────────

def _build_samples(
    data_root: Path,
    entries: list,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a flat metadata DataFrame from a list of split entries.

    Columns:
        take_name, video_path, gaze_frame_num, video_frame_idx,
        gaze_x_native, gaze_y_native, gaze_x, gaze_y

    Scans each take's eye gaze CSV (personalized preferred, general as fallback).
    Results are cached to pickle if cache_path is provided.
    """
    if cache_path is not None and cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Build lookup: take_name → root_dir
    with open(data_root / 'takes.json') as f:
        all_takes = json.load(f)
    name_to_root = {t['take_name']: t['root_dir'] for t in all_takes}

    records  = []
    n_skipped = 0

    for entry in entries:
        take_name = entry['take_name']
        root_dir  = name_to_root.get(take_name)
        if root_dir is None:
            n_skipped += 1
            continue

        take_dir   = data_root / root_dir
        base_video_dir = take_dir / 'frame_aligned_videos' / 'downscaled' / '448'
        video_path = base_video_dir / VIDEO_STREAM
        if not video_path.exists():
            matches = sorted(base_video_dir.glob("aria*_214-1.mp4"))
            video_path = matches[0] if matches else video_path

        # Try gaze CSV filenames in priority order:
        #   1. personalized (calibrated per-participant — preferred)
        #   2. general      (fallback: ~31% of takes lack personalized)
        #   3. legacy name used in some dataset releases
        gaze_csv = None
        for _name in ('personalized_eye_gaze_2d.csv',
                      'general_eye_gaze_2d.csv',
                      'personalized.csv'):
            _p = take_dir / 'eye_gaze' / _name
            if _p.exists():
                gaze_csv = _p
                break

        if gaze_csv is None or not video_path.exists():
            n_skipped += 1
            continue

        try:
            df = pd.read_csv(gaze_csv)
        except Exception:
            n_skipped += 1
            continue

        # ── Normalise column names ──────────────────────────────────────────
        # EgoExo-4D ships 'frame_number' (official) but some builds use 'frame_num'.
        # Rename to a canonical 'frame_num' for internal use.
        col_map = {}
        if 'frame_number' in df.columns and 'frame_num' not in df.columns:
            col_map['frame_number'] = 'frame_num'
        df = df.rename(columns=col_map)

        # Some releases use 'x_norm'/'y_norm' (normalised 0–1) instead of 'x'/'y'.
        if 'x_norm' in df.columns and 'x' not in df.columns:
            df = df.rename(columns={'x_norm': 'x', 'y_norm': 'y'})

        required = {'frame_num', 'x', 'y'}
        if not required.issubset(df.columns):
            # Log and skip — tells us which file is unexpected
            print(f"  [metadata] Unexpected CSV columns in {gaze_csv.name}: "
                  f"{list(df.columns)}  (expected {required})")
            n_skipped += 1
            continue

        # ── Drop rows with missing / empty gaze coords ─────────────────────
        valid_mask = df['x'].notna() & (df['x'].astype(str).str.strip() != '')
        df = df[valid_mask].copy()
        if df.empty:
            n_skipped += 1
            continue

        # ── Handle normalised (0–1) vs native (0–1408) coordinates ────────
        # If median x is in [0, 1] the coords are normalised → scale up first.
        x_vals = df['x'].astype(float)
        if x_vals.median() <= 1.0:
            # Normalised: multiply by native size to get pixel coords
            df['x'] = (x_vals * ARIA_NATIVE_SIZE).clip(0, ARIA_NATIVE_SIZE - 1)
            df['y'] = (df['y'].astype(float) * ARIA_NATIVE_SIZE).clip(
                0, ARIA_NATIVE_SIZE - 1
            )
        else:
            df['x'] = x_vals.clip(0, ARIA_NATIVE_SIZE - 1)
            df['y'] = df['y'].astype(float).clip(0, ARIA_NATIVE_SIZE - 1)

        # Scale native coords → TARGET_SIZE (300×300)
        df['gaze_x'] = (df['x'] * SCALE_FACTOR_X).clip(0, TARGET_SIZE[0] - 1)
        df['gaze_y'] = (df['y'] * SCALE_FACTOR_Y).clip(0, TARGET_SIZE[1] - 1)

        vp = str(video_path)
        for _, row in df.iterrows():
            gfn = int(row['frame_num'])
            records.append({
                'take_name':       take_name,
                'video_path':      vp,
                'gaze_frame_num':  gfn,
                'video_frame_idx': gfn * GAZE_SKIP,
                'gaze_x_native':   float(row['x']),
                'gaze_y_native':   float(row['y']),
                'gaze_x':          float(row['gaze_x']),
                'gaze_y':          float(row['gaze_y']),
            })

    if n_skipped:
        print(f"  [metadata] Skipped {n_skipped} takes (missing CSV / video)")

    df_out = pd.DataFrame(records)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(df_out, f)

    return df_out


# ── I-DT fixation detection ────────────────────────────────────────────────────

def _compute_idt_fixation(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    window: int   = IDT_WINDOW,
    threshold: float = IDT_DISPERSION_THRESHOLD,
) -> tuple:
    """
    I-DT fixation detection matching GazeLite.compute_fixation_state(method='idt').

    For each frame t, looks at the PAST window [t-window : t] (excluding t itself,
    exactly as gaze_history is sliced in train_stage3_epoch) and computes:

        dispersion = (max_x − min_x) + (max_y − min_y)

    Classified as fixation (1) if dispersion < threshold, saccade (0) otherwise.
    Fewer than 2 points in the window → fixation by fallback (matches model).

    Args:
        gaze_x, gaze_y: Coords in TARGET_SIZE (300px) space
        window:         How many past frames to look back (default: IDT_WINDOW=7)
        threshold:      Dispersion threshold in px  (default: IDT_DISPERSION_THRESHOLD=8.0)

    Returns:
        fixation:      int32 array  (1 = fixation, 0 = saccade)
        boundary_flag: bool array   (True at fixation→saccade transitions
                                     and at the last fixation frame)
    """
    n = len(gaze_x)
    fixation = np.zeros(n, dtype=np.int32)

    for t in range(n):
        # Past window only — mirrors `range(hist_start, t)` in train_stage3_epoch
        w_start = max(0, t - window)
        wx = gaze_x[w_start:t]
        wy = gaze_y[w_start:t]

        if len(wx) < 2:
            # Not enough history → treat as fixation (matches model fallback)
            fixation[t] = 1
            continue

        dispersion = (wx.max() - wx.min()) + (wy.max() - wy.min())
        fixation[t] = 1 if dispersion < threshold else 0

    boundary_flag = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if fixation[i] == 1 and fixation[i + 1] == 0:
            boundary_flag[i] = True
    if n > 0 and fixation[-1] == 1:
        boundary_flag[-1] = True

    return fixation, boundary_flag


# ── Stage 1 dataset ────────────────────────────────────────────────────────────

class EgoExo4DDataset(Dataset):
    """
    Stage 1 dataset: frame pairs (t, t−1) with gaze heatmap.

    frame_t         = video frame at gaze_frame_num × 3
    frame_t_minus_1 = video frame at gaze_frame_num × 3 − 1  (1 video frame back)
    heatmap_t       = Gaussian heatmap at TARGET_SIZE
    gaze_t          = [gaze_x, gaze_y] in TARGET_SIZE (300px) space
    gaze_t_minus_1  = gaze coords at previous 10fps frame (center fallback if missing)

    No fixation_t — velocity-based detection is used at runtime
    by GazeLite.compute_fixation_state().
    """

    def __init__(
        self,
        data_root: str,
        splits_json: str,
        split: str = 'train',
        option: str = 'b',
        transform=None,
        sample_fraction: float = 1.0,
        gt_sigma: float | None = None,
        cache_dir: str | None = None,
        frames_dir: str | None = None,
    ):
        self.data_root   = Path(data_root)
        self.splits_json = Path(splits_json)
        self.split       = split
        self.transform   = transform
        self.gt_sigma    = gt_sigma if gt_sigma is not None else GAUSSIAN_SIGMA

        entries = _load_split_entries(self.splits_json, option, split)

        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f'samples_{option}_{split}.pkl'

        print(f"[EgoExo4DDataset] Building metadata ({split}, option {option})...")
        self.metadata = _build_samples(self.data_root, entries, cache_path)

        # Drop first frame of each take (no prior video frame available)
        self.metadata = self.metadata[
            self.metadata['video_frame_idx'] > 0
        ].reset_index(drop=True)

        # Drop samples where either required frame is missing on disk.
        # Handles boundary frames where the gaze CSV extends one entry past
        # the last extracted video frame (off-by-one at end of take).
        if frames_dir is not None:
            frames_dir_path = Path(frames_dir)
            def _both_frames_exist(row):
                t  = frames_dir_path / row['take_name'] / f"frame_{int(row['video_frame_idx']):06d}.jpg"
                tm = frames_dir_path / row['take_name'] / f"frame_{int(row['video_frame_idx'])-1:06d}.jpg"
                return t.exists() and tm.exists()
            before = len(self.metadata)
            mask = self.metadata.apply(_both_frames_exist, axis=1)
            self.metadata = self.metadata[mask].reset_index(drop=True)
            dropped = before - len(self.metadata)
            if dropped:
                print(f"[EgoExo4DDataset] Dropped {dropped} samples with missing frames on disk")

        if sample_fraction < 1.0:
            n = int(len(self.metadata) * sample_fraction)
            self.metadata = (
                self.metadata.sample(n=n, random_state=42).reset_index(drop=True)
            )

        # Lookup for previous gaze label (keyed by take_name, gaze_frame_num)
        self._gaze_lookup = {
            (row['take_name'], row['gaze_frame_num']): (row['gaze_x'], row['gaze_y'])
            for _, row in self.metadata.iterrows()
        }

        self._video_cache = VideoCache(max_handles=10, frames_dir=frames_dir)

        mode = f"pre-extracted JPGs ({frames_dir})" if frames_dir else "on-the-fly mp4 decoding"
        print(f"[EgoExo4DDataset] {len(self.metadata)} samples "
              f"({split}, option {option})  |  {mode}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row             = self.metadata.iloc[idx]
        video_path      = row['video_path']
        vid_idx         = int(row['video_frame_idx'])
        gaze_frame_num  = int(row['gaze_frame_num'])
        take_name       = row['take_name']

        # Load frame pair
        frame_t         = self._load_frame(video_path, vid_idx, take_name)
        frame_t_minus_1 = self._load_frame(video_path, vid_idx - 1, take_name)

        # Heatmap from native gaze coordinates
        heatmap = create_gaze_heatmap(
            row['gaze_x_native'], row['gaze_y_native'], sigma=self.gt_sigma
        )
        heatmap_t = torch.from_numpy(heatmap).float()

        # Gaze in TARGET_SIZE space
        gaze_t = torch.tensor([row['gaze_x'], row['gaze_y']], dtype=torch.float32)

        # Previous gaze (fallback to image centre if unavailable)
        prev = self._gaze_lookup.get(
            (row['take_name'], gaze_frame_num - 1),
            (TARGET_SIZE[0] / 2.0, TARGET_SIZE[1] / 2.0),
        )
        gaze_t_minus_1 = torch.tensor(list(prev), dtype=torch.float32)

        return {
            'frame_t':         frame_t,
            'frame_t_minus_1': frame_t_minus_1,
            'heatmap_t':       heatmap_t,
            'gaze_t':          gaze_t,
            'gaze_t_minus_1':  gaze_t_minus_1,
        }

    def _load_frame(self, video_path: str, frame_idx: int,
                    take_name: str | None = None) -> torch.Tensor:
        frame = self._video_cache.get_frame(video_path, frame_idx, take_name)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        if self.transform:
            t = self.transform(t)
        return t

    def __del__(self):
        if hasattr(self, '_video_cache'):
            self._video_cache.release_all()


# ── Stage 2 / 3 dataset ───────────────────────────────────────────────────────

class EgoExo4DSequenceDataset(Dataset):
    """
    Stage 2/3 dataset: contiguous gaze sequences per take.

    Sequences are capped at max_seq_len to enable truncated BPTT.
    Ego-Exo-4D takes can span 1200–1800 gaze frames; without truncation
    Stage 3 would be ~20× slower than on EGTEA.

    Fixation labels and boundary flags are pre-computed via I-DT (dispersion
    threshold, matching GazeLite inference) — no ground-truth labels in Ego-Exo-4D.

    Each item:
        frames          [T, 3, H, W]
        gaze_coords     [T, 2]       float in TARGET_SIZE (300px) space
        fixation_labels [T, 1]       float  (1.0 = fixation, 0.0 = saccade)
        boundary_flags  [T, 1]       bool   (fixation→saccade transitions)
        seq_len         int
        take_name       str
    """

    def __init__(
        self,
        data_root: str,
        splits_json: str,
        split: str = 'train',
        option: str = 'b',
        transform=None,
        min_seq_len: int = 3,
        max_seq_len: int = 64,
        sample_fraction: float = 1.0,
        cache_dir: str | None = None,
        frames_dir: str | None = None,
    ):
        self.data_root   = Path(data_root)
        self.splits_json = Path(splits_json)
        self.split       = split
        self.transform   = transform
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        entries = _load_split_entries(self.splits_json, option, split)

        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f'samples_{option}_{split}.pkl'

        print(f"[EgoExo4DSequenceDataset] Building metadata ({split}, option {option})...")
        metadata = _build_samples(self.data_root, entries, cache_path)

        self.sequences = self._build_sequences(metadata)

        # Drop sequences that contain any missing frames on disk
        if frames_dir is not None:
            frames_root = Path(frames_dir)
            before = len(self.sequences)
            valid = []
            for seq in self.sequences:
                take_dir = frames_root / seq['take_name']
                missing = False
                for vf in seq['video_frames']:
                    vf = int(vf)
                    if not (take_dir / f"frame_{vf:06d}.jpg").exists():
                        missing = True
                        break
                    if vf > 0 and not (take_dir / f"frame_{vf-1:06d}.jpg").exists():
                        missing = True
                        break
                if not missing:
                    valid.append(seq)
            self.sequences = valid
            dropped = before - len(self.sequences)
            if dropped:
                print(f"[EgoExo4DSequenceDataset] Dropped {dropped} sequences with missing frames on disk")

        if sample_fraction < 1.0:
            n   = int(len(self.sequences) * sample_fraction)
            idx = np.random.RandomState(42).choice(len(self.sequences), n, replace=False)
            self.sequences = [self.sequences[i] for i in sorted(idx)]

        self._video_cache = VideoCache(max_handles=10, frames_dir=frames_dir)

        n_frames = sum(len(s['gaze_x']) for s in self.sequences)
        n_fix    = sum(
            int(s['fixation'].sum()) for s in self.sequences
        )
        print(
            f"[EgoExo4DSequenceDataset] {len(self.sequences)} sequences, "
            f"{n_frames} frames  ({split}, option {option}, max_len={max_seq_len})"
        )
        print(
            f"  Fixation: {n_fix} ({100*n_fix/max(n_frames,1):.1f}%)  "
            f"Saccade: {n_frames-n_fix} ({100*(n_frames-n_fix)/max(n_frames,1):.1f}%)"
        )

    def _build_sequences(self, metadata: pd.DataFrame) -> list:
        """
        Split each take into contiguous chunks, then sub-divide at max_seq_len
        for truncated BPTT.
        """
        sequences = []

        for take_name, take_df in metadata.groupby('take_name'):
            take_df    = take_df.sort_values('gaze_frame_num').reset_index(drop=True)
            video_path = take_df.iloc[0]['video_path']

            gaze_frames = take_df['gaze_frame_num'].values
            gaze_x      = take_df['gaze_x'].values.astype(np.float32)
            gaze_y      = take_df['gaze_y'].values.astype(np.float32)
            vid_frames  = take_df['video_frame_idx'].values

            # Identify contiguous chunks (gap > 1 gaze frame = discontinuity)
            chunks = []
            start  = 0
            for i in range(1, len(gaze_frames)):
                if gaze_frames[i] - gaze_frames[i - 1] > 1:
                    if i - start >= self.min_seq_len:
                        chunks.append((start, i))
                    start = i
            if len(gaze_frames) - start >= self.min_seq_len:
                chunks.append((start, len(gaze_frames)))

            for c_start, c_end in chunks:
                # Sub-divide long chunks into max_seq_len windows
                for seg_start in range(c_start, c_end, self.max_seq_len):
                    seg_end = min(seg_start + self.max_seq_len, c_end)

                    if seg_end - seg_start < self.min_seq_len:
                        continue

                    sx = gaze_x[seg_start:seg_end]
                    sy = gaze_y[seg_start:seg_end]
                    sv = vid_frames[seg_start:seg_end]

                    fixation, boundary = _compute_idt_fixation(sx, sy)

                    sequences.append({
                        'take_name':    take_name,
                        'video_path':   video_path,
                        'video_frames': sv,
                        'gaze_x':       sx,
                        'gaze_y':       sy,
                        'fixation':     fixation,
                        'boundary':     boundary,
                    })

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        T   = len(seq['gaze_x'])

        frames          = []
        gaze_coords     = []
        fixation_labels = []
        boundary_flags  = []

        for t in range(T):
            vid_frame = int(seq['video_frames'][t])
            frame = self._load_frame(seq['video_path'], vid_frame, seq['take_name'])
            frames.append(frame)

            gaze_coords.append(
                torch.tensor([seq['gaze_x'][t], seq['gaze_y'][t]], dtype=torch.float32)
            )
            fixation_labels.append(
                torch.tensor([seq['fixation'][t]], dtype=torch.float32)
            )
            boundary_flags.append(
                torch.tensor([seq['boundary'][t]], dtype=torch.bool)
            )

        return {
            'frames':          torch.stack(frames),           # [T, 3, H, W]
            'gaze_coords':     torch.stack(gaze_coords),      # [T, 2]
            'fixation_labels': torch.stack(fixation_labels),  # [T, 1]
            'boundary_flags':  torch.stack(boundary_flags),   # [T, 1]
            'seq_len':         T,
            'take_name':       seq['take_name'],
        }

    def _load_frame(self, video_path: str, frame_idx: int,
                    take_name: str | None = None) -> torch.Tensor:
        frame = self._video_cache.get_frame(video_path, frame_idx, take_name)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        if self.transform:
            t = self.transform(t)
        return t

    def __del__(self):
        if hasattr(self, '_video_cache'):
            self._video_cache.release_all()


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transform():
    """Returns ImageNet normalisation transform."""
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


# ── Factory functions ──────────────────────────────────────────────────────────

def create_dataloaders(
    data_root: str,
    splits_json: str,
    option: str = 'b',
    batch_size: int = 8,
    num_workers: int = 8,
    sample_fraction: float = 1.0,
    val_sample_fraction: float | None = None,
    gt_sigma: float | None = None,
    cache_dir: str | None = None,
    frames_dir: str | None = None,
):
    """Create Stage 1 train and val DataLoaders.

    val_sample_fraction: fraction of val set to use during training validation.
    Defaults to sample_fraction if not set. Use a small value (e.g. 0.2) to
    keep validation fast — the full val set is evaluated by evaluate.py.
    """
    transform = get_transform()
    _val_frac = val_sample_fraction if val_sample_fraction is not None else sample_fraction

    train_ds = EgoExo4DDataset(
        data_root=data_root, splits_json=splits_json,
        split='train', option=option,
        transform=transform, sample_fraction=sample_fraction,
        gt_sigma=gt_sigma, cache_dir=cache_dir, frames_dir=frames_dir,
    )
    val_ds = EgoExo4DDataset(
        data_root=data_root, splits_json=splits_json,
        split='val', option=option,
        transform=transform, sample_fraction=_val_frac,
        gt_sigma=gt_sigma, cache_dir=cache_dir, frames_dir=frames_dir,
    )

    # persistent_workers=False: workers are torn down after each epoch (~10s
    # respawn cost) but this prevents /dev/shm exhaustion from concurrent
    # train+val worker pools causing sporadic segfaults during validation.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader


def create_sequence_dataloader(
    data_root: str,
    splits_json: str,
    split: str = 'train',
    option: str = 'b',
    batch_size: int = 1,
    num_workers: int = 8,
    sample_fraction: float = 1.0,
    max_seq_len: int = 64,
    cache_dir: str | None = None,
    frames_dir: str | None = None,
):
    """Create Stage 2/3 sequence DataLoader for one split."""
    transform = get_transform()

    dataset = EgoExo4DSequenceDataset(
        data_root=data_root, splits_json=splits_json,
        split=split, option=option,
        transform=transform,
        min_seq_len=3, max_seq_len=max_seq_len,
        sample_fraction=sample_fraction, cache_dir=cache_dir,
        frames_dir=frames_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
    )
    return loader
