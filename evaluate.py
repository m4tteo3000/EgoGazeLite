"""
Evaluation script for trained GazeLite models on Ego-Exo-4D.

Evaluates on SEQUENCES (matching Huang et al.) with:
  - LSTM hidden state carried through each sequence
  - Real gaze history built up frame by frame for I-DT fixation detection
  - Metrics computed on ALL frames (matching Huang)
  - Both G_s (saliency only) and G_t (full model, I-DT) evaluated

Metrics: AUC, NSS, AAE, Pixel Distance.

AAE note:
  --camera_fov 78  → Aria RGB horizontal FOV (Ego-Exo-4D, default)
  --camera_fov 60  → EGTEA default — use only for cross-dataset comparison

Results are printed, logged to W&B, and saved to JSON.

Usage:
    python evaluate.py \\
        --checkpoint experiments/full_run_egoexo_option_b_cotrain_v1/checkpoint_final.pt \\
        --data_root  /path/to/EgoExo4D \\
        --splits_json configs/egoexo4d_splits.json \\
        --split       val \\
        --option      b \\
        --cache_dir   /path/to/EgoExo4D/.cache
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.gaze_lite import GazeLite, FIXATION_IDT_WINDOW
from src.datasets import create_sequence_dataloader, GAUSSIAN_SIGMA
from src.metrics import (
    compute_auc,
    compute_nss,
    compute_aae,
    compute_pixel_distance,
    count_parameters,
    get_model_size_mb,
    compute_inference_time,
    make_gaussian_kernel,
    create_heatmap_gpu,
)

# I-DT window size — imported from gaze_lite.py to stay in sync with training
IDT_WINDOW = FIXATION_IDT_WINDOW


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint, respecting saved config."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    fusion_mode       = config.get('stage3', {}).get('fusion_mode', 'residual')
    center_bias_sigma = config.get('stage1', {}).get('center_bias_sigma', 0.4)

    model = GazeLite(
        pretrained_backbone=False,
        use_skip_connections=True,
        use_center_bias=True,
        center_bias_sigma=center_bias_sigma,
        fusion_mode=fusion_mode,
    )

    # Strip torch.compile / DDP prefixes so we can load uncompiled-into-uncompiled
    # (training was run with torch_compile=true, so saved keys carry _orig_mod.).
    def _strip(k):
        return k.replace('_orig_mod.', '').replace('module.', '')

    model_state      = model.state_dict()
    pretrained_state = checkpoint['model_state_dict']
    stripped_to_model = {_strip(k): k for k in model_state}

    filtered = {}
    for ck_key, ck_val in pretrained_state.items():
        m_key = stripped_to_model.get(_strip(ck_key))
        if m_key is not None and ck_val.shape == model_state[m_key].shape:
            filtered[m_key] = ck_val

    missing = len(model_state) - len(filtered)
    skipped = len(pretrained_state) - len(filtered)
    if missing > 0 or skipped > 0:
        print(f"  load: {len(filtered)} loaded, {skipped} ckpt keys skipped, {missing} model keys unfilled")
    if len(filtered) == 0:
        raise RuntimeError(
            f"Loaded 0/{len(pretrained_state)} keys. Sample ckpt key: "
            f"{next(iter(pretrained_state))!r}, sample model key: {next(iter(model_state))!r}"
        )

    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Stage: {checkpoint.get('stage', '?')}  |  "
          f"Epoch: {checkpoint.get('epoch', '?')}  |  "
          f"Fusion: {fusion_mode}  |  CB sigma: {center_bias_sigma}")
    return model, checkpoint


# =============================================================================
# Metric Accumulator
# =============================================================================

class MetricAccumulator:
    """Accumulates per-frame metric scalars. Constant RAM."""

    def __init__(self, fov_degrees=None):
        self.fov_degrees = fov_degrees   # passed through to compute_aae
        self.reset()

    def reset(self):
        self.auc            = []
        self.nss            = []
        self.aae            = []
        self.pixel_distance = []

    def update(self, pred, target):
        """pred/target: [B, 1, H, W] on CPU."""
        self.auc.append(compute_auc(pred, target))
        self.nss.append(compute_nss(pred, target))
        self.aae.append(compute_aae(pred, target, fov_degrees=self.fov_degrees))
        self.pixel_distance.append(compute_pixel_distance(pred, target))

    def compute(self):
        return {
            'auc':            float(np.mean(self.auc)),
            'nss':            float(np.mean(self.nss)),
            'aae':            float(np.mean(self.aae)),
            'pixel_distance': float(np.mean(self.pixel_distance)),
            'n_frames':       len(self.auc),
        }


# =============================================================================
# Sequence Evaluation
# =============================================================================

def evaluate_sequences(model, loader, device, gaussian_kernel, fov_degrees=None,
                       take_to_domain=None):
    """
    Evaluate on full sequences, matching Huang et al.:
      - LSTM hidden state carried through each sequence, reset between sequences
      - Real gaze history accumulated for I-DT (GT gaze coords from dataset)
      - Metrics computed on every frame

    Args:
        take_to_domain: Optional dict {take_name: domain} — when provided,
                        per-domain MetricAccumulators are built on the fly.

    Returns:
        metrics_gs:      overall saliency-only metrics
        metrics_gt:      overall full-model metrics
        per_domain_gs:   dict {domain: metrics} (or {} if take_to_domain is None)
        per_domain_gt:   dict {domain: metrics} (or {} if take_to_domain is None)
    """
    print("\nRunning sequence evaluation...")

    acc_gs = MetricAccumulator(fov_degrees=fov_degrees)
    acc_gt = MetricAccumulator(fov_degrees=fov_degrees)

    # Per-domain accumulators, created lazily as we see each domain
    dom_gs: dict = {}
    dom_gt: dict = {}

    def _dom_acc(domain):
        if domain not in dom_gs:
            dom_gs[domain] = MetricAccumulator(fov_degrees=fov_degrees)
            dom_gt[domain] = MetricAccumulator(fov_degrees=fov_degrees)
        return dom_gs[domain], dom_gt[domain]

    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating sequences"):
            frames      = batch['frames'].to(device)       # [1, T, 3, H, W]
            gaze_coords = batch['gaze_coords'].to(device)  # [1, T, 2]
            seq_len     = batch['seq_len']                 # int

            # Resolve domain for this sequence (batch_size=1 → take from first)
            domain = None
            if take_to_domain is not None:
                tn = batch['take_name']
                take_name = tn[0] if isinstance(tn, (list, tuple)) else tn
                domain = take_to_domain.get(take_name, 'UNKNOWN')

            hidden       = None
            gaze_history = []

            for t in range(1, seq_len[0]):
                frame_t         = frames[:, t]
                frame_t_minus_1 = frames[:, t - 1]
                gaze_t          = gaze_coords[:, t]
                gaze_t_minus_1  = gaze_coords[:, t - 1]

                # Build real gaze history for I-DT
                gaze_history.append(gaze_t_minus_1)
                if len(gaze_history) > IDT_WINDOW:
                    gaze_history.pop(0)

                # GT heatmap on GPU
                gt_heatmap = create_heatmap_gpu(gaze_t, gaussian_kernel, size=300)

                # --- Saliency only (G_s) ---
                g_s      = model.forward_saliency_only(frame_t, frame_t_minus_1)
                g_s_prob = torch.sigmoid(g_s) if model.saliency_decoder.return_logits else g_s

                # --- Full model (G_t) with real I-DT gaze history ---
                g_t_out, _, hidden, _ = model(
                    frame_t, frame_t_minus_1, gaze_t_minus_1,
                    hidden=hidden,
                    gaze_history=gaze_history,
                    fixation_method='idt',
                )
                # model() was called with return_logits=False (default),
                # so fusion already applied sigmoid — do NOT apply it again.
                g_t_prob = g_t_out

                # Detach hidden state to avoid backprop through sequences
                hidden = (hidden[0].detach(), hidden[1].detach())

                # Accumulate metrics on every frame (matching Huang)
                gs_cpu = g_s_prob.cpu()
                gt_cpu = g_t_prob.cpu()
                hm_cpu = gt_heatmap.cpu()

                acc_gs.update(gs_cpu, hm_cpu)
                acc_gt.update(gt_cpu, hm_cpu)

                if domain is not None:
                    d_gs, d_gt = _dom_acc(domain)
                    d_gs.update(gs_cpu, hm_cpu)
                    d_gt.update(gt_cpu, hm_cpu)

    per_domain_gs = {d: acc.compute() for d, acc in dom_gs.items()}
    per_domain_gt = {d: acc.compute() for d, acc in dom_gt.items()}

    return acc_gs.compute(), acc_gt.compute(), per_domain_gs, per_domain_gt


# =============================================================================
# Efficiency
# =============================================================================

def compute_efficiency_metrics(model, device):
    print("\nComputing efficiency metrics...")
    m = {}
    m['parameters']        = count_parameters(model)
    m['model_size_mb']     = get_model_size_mb(model)
    m['inference_time_ms'] = compute_inference_time(model, device=device, num_runs=50)
    m['fps']               = 1000.0 / m['inference_time_ms']
    return m


# =============================================================================
# Printing
# =============================================================================

def print_results(label, metrics, efficiency=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'AUC':<22} {metrics['auc']:.4f}")
    print(f"  {'NSS':<22} {metrics['nss']:.4f}")
    print(f"  {'AAE (deg)':<22} {metrics['aae']:.4f}")
    print(f"  {'Pixel Distance':<22} {metrics['pixel_distance']:.2f}px")
    print(f"  {'Frames evaluated':<22} {metrics['n_frames']:,}")
    if efficiency:
        print(f"\n  {'Parameters':<22} {efficiency['parameters']:,}")
        print(f"  {'Model Size':<22} {efficiency['model_size_mb']:.2f} MB")
        print(f"  {'Inference':<22} {efficiency['inference_time_ms']:.2f} ms  "
              f"({efficiency['fps']:.1f} FPS)")
    print(f"{'='*60}")


def print_comparison_table(metrics_gs, metrics_gt):
    keys = ['auc', 'nss', 'aae', 'pixel_distance']
    print(f"\n{'='*60}")
    print(f"  ABLATION: SALIENCY vs FULL MODEL")
    print(f"{'='*60}")
    print(f"  {'Metric':<22} {'G_s (saliency)':>16} {'G_t (full)':>12}")
    print(f"  {'-'*50}")
    for k in keys:
        print(f"  {k:<22} {metrics_gs[k]:>16.4f} {metrics_gt[k]:>12.4f}")
    print(f"{'='*60}")


def print_per_domain_table(per_domain_gs, per_domain_gt):
    """Print one row per domain with G_s and G_t side-by-side."""
    if not per_domain_gs:
        return
    domains = sorted(per_domain_gs.keys())
    print(f"\n{'='*96}")
    print(f"  PER-DOMAIN BREAKDOWN")
    print(f"{'='*96}")
    header = (f"  {'Domain':<22} {'N frames':>10}  "
              f"{'AUC (s/t)':>14}  {'NSS (s/t)':>14}  "
              f"{'AAE (s/t)':>14}  {'PxDist (s/t)':>16}")
    print(header)
    print(f"  {'-'*92}")
    for d in domains:
        gs = per_domain_gs[d]
        gt = per_domain_gt[d]
        print(
            f"  {d:<22} {gs['n_frames']:>10,}  "
            f"{gs['auc']:>6.3f}/{gt['auc']:<6.3f}  "
            f"{gs['nss']:>6.3f}/{gt['nss']:<6.3f}  "
            f"{gs['aae']:>6.2f}/{gt['aae']:<6.2f}  "
            f"{gs['pixel_distance']:>7.2f}/{gt['pixel_distance']:<7.2f}"
        )
    print(f"{'='*96}")


# =============================================================================
# W&B + Save
# =============================================================================

def log_to_wandb(metrics_gs, metrics_gt, efficiency,
                 per_domain_gs=None, per_domain_gt=None):
    wandb.log({f'saliency_only/{k}': v for k, v in metrics_gs.items()})
    wandb.log({f'full_model/{k}':    v for k, v in metrics_gt.items()})
    wandb.log({f'efficiency/{k}':    v for k, v in efficiency.items()})

    table = wandb.Table(columns=['Path', 'AUC', 'NSS', 'AAE', 'Pixel Dist', 'Frames'])
    for lbl, m in [('G_s (saliency)', metrics_gs), ('G_t (full model)', metrics_gt)]:
        table.add_data(lbl, round(m['auc'], 4), round(m['nss'], 4),
                       round(m['aae'], 4), round(m['pixel_distance'], 2),
                       m['n_frames'])
    wandb.log({'ablation_table': table})

    # Per-domain: log each metric keyed by domain, plus a summary table
    if per_domain_gs:
        for d, m in per_domain_gs.items():
            # W&B-safe key: replace spaces / slashes
            key = d.replace(' ', '_').replace('/', '_')
            for k, v in m.items():
                wandb.log({f'per_domain/{key}/saliency_only/{k}': v})
        for d, m in per_domain_gt.items():
            key = d.replace(' ', '_').replace('/', '_')
            for k, v in m.items():
                wandb.log({f'per_domain/{key}/full_model/{k}': v})

        dtable = wandb.Table(columns=[
            'Domain', 'N frames',
            'AUC (G_s)', 'AUC (G_t)',
            'NSS (G_s)', 'NSS (G_t)',
            'AAE (G_s)', 'AAE (G_t)',
            'PxDist (G_s)', 'PxDist (G_t)',
        ])
        for d in sorted(per_domain_gs.keys()):
            gs = per_domain_gs[d]
            gt = per_domain_gt[d]
            dtable.add_data(
                d, gs['n_frames'],
                round(gs['auc'], 4),            round(gt['auc'], 4),
                round(gs['nss'], 4),            round(gt['nss'], 4),
                round(gs['aae'], 4),            round(gt['aae'], 4),
                round(gs['pixel_distance'], 2), round(gt['pixel_distance'], 2),
            )
        wandb.log({'per_domain_table': dtable})


def save_results(metrics_gs, metrics_gt, efficiency, output_path,
                 checkpoint_path, fov_degrees,
                 per_domain_gs=None, per_domain_gt=None,
                 split=None, option=None):
    results = {
        'checkpoint':    checkpoint_path,
        'timestamp':     datetime.now().isoformat(),
        'camera_fov':    fov_degrees,
        'split':         split,
        'option':        option,
        'evaluation':    'sequence-based, all frames, I-DT fixation with real gaze history',
        'saliency_only': metrics_gs,
        'full_model':    metrics_gt,
        'efficiency':    {k: float(v) for k, v in efficiency.items()},
    }
    if per_domain_gs:
        results['per_domain'] = {
            d: {
                'saliency_only': per_domain_gs[d],
                'full_model':    per_domain_gt[d],
            }
            for d in sorted(per_domain_gs.keys())
        }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GazeLite model on Ego-Exo-4D sequences'
    )
    # Checkpoint + output
    parser.add_argument('--checkpoint',      type=str,   required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--output',          type=str,   default=None,
                        help='Path to save JSON results (auto-generated if omitted)')
    # Data
    parser.add_argument('--data_root',       type=str,   required=True,
                        help='Ego-Exo-4D root directory on disk')
    parser.add_argument('--splits_json',     type=str,
                        default='configs/egoexo4d_splits.json',
                        help='Path to egoexo4d_splits.json')
    parser.add_argument('--split',           type=str,   default='val',
                        choices=['train', 'val', 'downstream_test', 'generalization_test'],
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--option',          type=str,   default='b',
                        choices=['a', 'b'],
                        help='Data option tier: a (~28h) or b (~68h) (default: b)')
    parser.add_argument('--cache_dir',       type=str,   default=None,
                        help='Directory for metadata pickle cache (speeds up init)')
    parser.add_argument('--frames_dir',      type=str,   default=None,
                        help='Directory of pre-extracted JPG frames (fast path). '
                             'If omitted, mp4s are decoded on the fly.')
    parser.add_argument('--bptt_window',     type=int,   default=64,
                        help='Max sequence length for evaluation (default: 64)')
    parser.add_argument('--no_per_domain',   action='store_true',
                        help='Disable per-domain metric breakdown. '
                             'Per-domain is ON by default — it adds a few small '
                             'accumulators and has no effect on overall numbers.')
    # Metric
    parser.add_argument('--camera_fov',      type=float, default=78.0,
                        help='Horizontal camera FOV in degrees for AAE metric. '
                             '78 = Aria RGB (Ego-Exo-4D default). '
                             '60 = EGTEA (for cross-dataset comparison only).')
    # Misc
    parser.add_argument('--num_workers',     type=int,   default=8)
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    parser.add_argument('--device',          type=str,   default=None)
    parser.add_argument('--wandb_project',   type=str,   default='ego-exo-gaze')
    parser.add_argument('--wandb_run',       type=str,   default=None)
    parser.add_argument('--no_wandb',        action='store_true')
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Camera FOV for AAE: {args.camera_fov}°")

    # W&B
    if not args.no_wandb:
        run_name = args.wandb_run or \
            f"eval_{args.split}_{os.path.basename(args.checkpoint).replace('.pt', '')}"
        wandb.init(project=args.wandb_project, name=run_name,
                   job_type='evaluation', config=vars(args))
        print(f"W&B run: {wandb.run.name}")

    # Model
    model, _ = load_model(args.checkpoint, device)

    # Gaussian kernel for GPU heatmap generation
    gaussian_kernel = make_gaussian_kernel(GAUSSIAN_SIGMA).to(device)

    # Sequence dataloader (Ego-Exo-4D)
    print(f"\nLoading {args.split} sequences "
          f"(option={args.option}, sample_fraction={args.sample_fraction})...")
    test_loader = create_sequence_dataloader(
        data_root        = args.data_root,
        splits_json      = args.splits_json,
        split            = args.split,
        option           = args.option,
        batch_size       = 1,
        num_workers      = args.num_workers,
        sample_fraction  = args.sample_fraction,
        max_seq_len      = args.bptt_window,
        cache_dir        = args.cache_dir,
        frames_dir       = args.frames_dir,
    )
    print(f"  Sequences: {len(test_loader.dataset)}")

    # Build take_name → domain map from splits JSON (for per-domain metrics)
    take_to_domain = None
    if not args.no_per_domain:
        with open(args.splits_json) as f:
            _split_data = json.load(f)
        take_to_domain = {
            e['take_name']: e.get('domain', 'UNKNOWN')
            for e in _split_data.get('takes', [])
        }
        gen_domain = _split_data.get('metadata', {}).get('generalization_domain')
        if gen_domain and args.split == 'generalization_test':
            print(f"  Generalization split — expecting held-out domain: {gen_domain}")

    # Evaluate
    metrics_gs, metrics_gt, per_domain_gs, per_domain_gt = evaluate_sequences(
        model, test_loader, device, gaussian_kernel,
        fov_degrees=args.camera_fov,
        take_to_domain=take_to_domain,
    )

    # Efficiency
    efficiency = compute_efficiency_metrics(model, device)

    # Print
    print_results("SALIENCY ONLY (G_s)", metrics_gs)
    print_results("FULL MODEL (G_t, I-DT)", metrics_gt, efficiency)
    print_comparison_table(metrics_gs, metrics_gt)
    print_per_domain_table(per_domain_gs, per_domain_gt)

    # W&B
    if not args.no_wandb:
        log_to_wandb(metrics_gs, metrics_gt, efficiency,
                     per_domain_gs=per_domain_gs, per_domain_gt=per_domain_gt)
        wandb.finish()

    # Save
    if args.output is None:
        ckpt_name   = os.path.basename(args.checkpoint).replace('.pt', '')
        args.output = (f"experiments/eval_{args.split}_{ckpt_name}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    save_results(metrics_gs, metrics_gt, efficiency,
                 args.output, args.checkpoint, args.camera_fov,
                 per_domain_gs=per_domain_gs, per_domain_gt=per_domain_gt,
                 split=args.split, option=args.option)
    print("\nDone!")


if __name__ == '__main__':
    main()
