"""
Staged training script for GazeLite.
Stage 1: Saliency decoder
Stage 2: Attention transition (LSTM)
Stage 3: Gated fusion

"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

from src.models.gaze_lite import GazeLite, FIXATION_IDT_WINDOW
from src.datasets import create_dataloaders, create_sequence_dataloader
from src.datasets import create_gaze_heatmap, SCALE_FACTOR_X, SCALE_FACTOR_Y, GAUSSIAN_SIGMA
from src.losses import SaliencyLoss, SaliencyLogitsLoss, SaliencyKLDLoss, AttentionTransitionLoss, FusionLoss
from src.metrics import (
    compute_auc, compute_nss, compute_aae, compute_pixel_distance,
    compute_distance_f1,
    count_parameters, get_model_size_mb,
    make_gaussian_kernel, create_heatmap_gpu,
)


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def setup_wandb(config, model):
    """Initialise W&B logging."""
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['run_name'],
            config=config
        )
        
        # Watch model for gradient and weight logging
        log_freq = config['wandb'].get('log_freq', 100)
        log_gradients = config['wandb'].get('log_gradients', False)
        log_weights = config['wandb'].get('log_weights', False)
        
        if log_gradients or log_weights:
            log_option = "all" if (log_gradients and log_weights) else ("gradients" if log_gradients else "parameters")
            wandb.watch(model, log=log_option, log_freq=log_freq)
            print(f"W&B watching model: {log_option} every {log_freq} batches")
        
        return True
    return False


def freeze_module(module):
    """
    Freeze a module: stop grad updates AND put it in eval mode.

    .eval() is critical: without it, BatchNorm layers continue to (a) use
    current-batch stats and (b) update running_mean / running_var every
    forward pass. requires_grad=False does NOT stop BN buffer updates.
    In staged training, that corrupts the "frozen" Stage 1/2 running stats
    during Stage 2/3 training and silently degrades the frozen path.
    """
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def set_frozen_modules_eval(model, stage):
    """
    Re-apply .eval() to frozen modules after a top-level model.train() call.

    model.train() recursively puts every submodule (including frozen ones)
    back in training mode, which re-enables BN stat updates. Call this at
    the top of each training epoch after model.train().
    """
    if stage == 2:
        frozen = [model.backbone, model.temporal_diff, model.saliency_decoder,
                  model.gated_fusion]
    elif stage == 3:
        # Only backbone / temporal_diff / saliency are guaranteed frozen.
        # Attention path may be co-trained (stage3.cotrain_attention=true), so
        # build the list dynamically from each submodule's actual state.
        frozen = [model.backbone, model.temporal_diff, model.saliency_decoder]
        for m in (model.channel_weight_extractor, model.lstm_gated,
                  model.attention_weight_app):
            if not any(p.requires_grad for p in m.parameters()):
                frozen.append(m)
    else:
        return
    for m in frozen:
        m.eval()


def unfreeze_module(module):
    """Unfreeze module parameters."""
    for param in module.parameters():
        param.requires_grad = True


def setup_model_for_stage(model, stage, config=None):
    """
    Freeze/unfreeze appropriate modules for each training stage.

    Stage 1 backbone freezing controlled by config['stage1']['backbone_freeze']:
        "none"             — all backbone layers trainable (default)
        "all"              — entire backbone frozen
        "all_except_last"  — only last block trainable

    Stage 3 co-training controlled by config['stage3']['cotrain_attention']:
        false (default) — only gated_fusion trainable
        true            — gated_fusion + attention path (channel_weight_extractor,
                          lstm_gated, attention_weight_app) all trainable.
                          Backbone / temporal_diff / saliency_decoder stay frozen.
                          Use lower lr for attention path via stage3.attention_lr.
    """
    if stage == 1:
        # Backbone freezing from config
        backbone_freeze = "none"
        if config is not None:
            backbone_freeze = config.get('stage1', {}).get('backbone_freeze', 'none')
        
        if backbone_freeze == "none":
            unfreeze_module(model.backbone)
        elif backbone_freeze == "all":
            freeze_module(model.backbone)
        elif backbone_freeze == "all_except_last":
            freeze_module(model.backbone)
            for param in model.backbone.backbone.blocks[-1].parameters():
                param.requires_grad = True
        
        unfreeze_module(model.temporal_diff)
        unfreeze_module(model.saliency_decoder)
        freeze_module(model.channel_weight_extractor)
        freeze_module(model.lstm_gated)
        freeze_module(model.attention_weight_app)
        freeze_module(model.gated_fusion)
        
    elif stage == 2:
        freeze_module(model.backbone)
        freeze_module(model.temporal_diff)
        freeze_module(model.saliency_decoder)
        unfreeze_module(model.channel_weight_extractor)
        unfreeze_module(model.lstm_gated)
        unfreeze_module(model.attention_weight_app)
        freeze_module(model.gated_fusion)
        
    elif stage == 3:
        freeze_module(model.backbone)
        freeze_module(model.temporal_diff)
        freeze_module(model.saliency_decoder)

        cotrain_attention = False
        if config is not None:
            cotrain_attention = config.get('stage3', {}).get('cotrain_attention', False)

        if cotrain_attention:
            unfreeze_module(model.channel_weight_extractor)
            unfreeze_module(model.lstm_gated)
            unfreeze_module(model.attention_weight_app)
            print("  Stage 3: co-training attention path + fusion")
        else:
            freeze_module(model.channel_weight_extractor)
            freeze_module(model.lstm_gated)
            freeze_module(model.attention_weight_app)

        unfreeze_module(model.gated_fusion)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: {trainable:,} / {total:,} parameters trainable")



# =============================================================================
# Training Functions
# =============================================================================

def train_stage1_epoch(model, loader, criterion, optimizer, device, config, epoch, global_step):
    """Train one epoch for Stage 1 (saliency decoder). AMP-enabled."""
    model.train()
    total_loss = 0
    log_freq = config['wandb'].get('log_freq', 50)
    use_amp = config.get('training', {}).get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    pbar = tqdm(loader, desc=f"Training Stage 1 Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        frame_t = batch['frame_t'].to(device)
        frame_t_minus_1 = batch['frame_t_minus_1'].to(device)
        heatmap_t = batch['heatmap_t'].to(device)
        gaze_t = batch['gaze_t'].to(device)
        
        with torch.autocast('cuda', enabled=use_amp):
            g_s = model.forward_saliency_only(frame_t, frame_t_minus_1)
            gt_heatmap = heatmap_t.unsqueeze(1)
            loss = criterion(g_s, gt_heatmap, gaze_t)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        global_step += 1
        
        if config['wandb']['enabled'] and batch_idx % log_freq == 0:
            with torch.no_grad():
                g_s_prob = torch.sigmoid(g_s) if model.saliency_decoder.return_logits else g_s
                batch_pixel_dist = compute_pixel_distance(g_s_prob, gt_heatmap)
            
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_pixel_dist': batch_pixel_dist,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/global_step': global_step,
            }, step=global_step)
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader), global_step


def validate_stage1(model, loader, criterion, device, config, epoch, log_predictions=False):
    """Validate Stage 1. Computes core metrics only (AUC, NSS, AAE, pixel dist)
    to keep validation fast. Extended metrics live in evaluate.py."""
    model.eval()
    total_loss = 0
    all_auc = []
    all_nss = []
    all_aae = []

    # For prediction logging
    sample_images = []
    sample_preds = []
    sample_gts = []
    num_samples = config['wandb'].get('num_prediction_samples', 4)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating Stage 1")):
            frame_t = batch['frame_t'].to(device)
            frame_t_minus_1 = batch['frame_t_minus_1'].to(device)
            heatmap_t = batch['heatmap_t'].to(device)
            gaze_t = batch['gaze_t'].to(device)

            g_s = model.forward_saliency_only(frame_t, frame_t_minus_1)
            g_s_prob = torch.sigmoid(g_s) if model.saliency_decoder.return_logits else g_s

            gt_heatmap = heatmap_t.unsqueeze(1)
            loss = criterion(g_s, gt_heatmap, gaze_t)
            total_loss += loss.item()

            # Move to CPU once, reuse for all metrics
            g_s_cpu  = g_s_prob.cpu()
            gt_cpu   = gt_heatmap.cpu()

            all_auc.append(compute_auc(g_s_cpu, gt_cpu))
            all_nss.append(compute_nss(g_s_cpu, gt_cpu))
            all_aae.append(compute_aae(g_s_cpu, gt_cpu, fov_degrees=config.get('camera_fov', None)))

            # Collect samples for visualization
            if log_predictions and len(sample_images) < num_samples:
                for i in range(min(g_s.shape[0], num_samples - len(sample_images))):
                    sample_images.append(frame_t[i].cpu())
                    sample_preds.append(g_s_prob[i, 0].cpu())
                    sample_gts.append(heatmap_t[i].cpu())

    metrics = {
        'loss': total_loss / len(loader),
        'auc':  float(np.mean(all_auc)),
        'nss':  float(np.mean(all_nss)),
        'aae':  float(np.mean(all_aae)),
    }
    
    # Log prediction visualizations
    if log_predictions and config['wandb']['enabled'] and len(sample_images) > 0:
        log_prediction_images(sample_images, sample_preds, sample_gts, epoch)
    
    return metrics


def log_prediction_images(images, preds, gts, epoch):
    """Log Stage 1 sample predictions to W&B (3-panel: frame, pred, GT)."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (img, pred, gt) in enumerate(zip(images, preds, gts)):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Frame")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred.numpy(), cmap='hot')
        axes[i, 1].set_title("Predicted Saliency")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(gt.numpy(), cmap='hot')
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    wandb.log({f"predictions/epoch_{epoch}": wandb.Image(fig)})
    plt.close(fig)


def log_center_bias_visualization(model, epoch):
    """Log the learned center bias as an image."""
    if hasattr(model.saliency_decoder, 'center_bias'):
        cb = model.saliency_decoder.center_bias.detach().cpu().squeeze().numpy()
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(cb, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"Learned Center Bias (Epoch {epoch})")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
        wandb.log({f"center_bias/epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)


# =============================================================================
# Stage 2: Training & Validation
# =============================================================================

def train_stage2_epoch(model, loader, attention_criterion, optimizer, device):
    """
    Train one epoch for Stage 2 (LSTM attention transition).

    Following Huang et al.: LSTM only sees fixation boundary frames.
    Sequence: [w_fixation1, w_fixation2, ...] — one per fixation period.
    Loss: predict w_{n+1} from w_n.

    Speedup: backbone + saliency decoder run once over all T frames as a
    single [B*T, C, H, W] batch (precompute_sequence_features), then only
    the cheap channel_weight_extractor + lstm_gated loop over boundaries.
    """
    model.train()
    # Keep frozen submodules in eval mode so their BN running stats aren't
    # corrupted by the large batched forward pass in precompute_sequence_features.
    set_frozen_modules_eval(model, stage=2)
    total_loss = 0
    n_boundary_steps = 0
    n_boundary_batches = 0

    pbar = tqdm(loader, desc="Training Stage 2")
    for batch in pbar:
        frames        = batch['frames'].to(device)         # [B, T, C, H, W]
        gaze_coords   = batch['gaze_coords'].to(device)
        boundary_flags = batch['boundary_flags'].to(device)
        seq_len        = batch['seq_len']
        T = seq_len[0]

        # ── Pre-compute all backbone features in one batched pass ─────────
        # Replaces T sequential _extract_features calls; ~10× fewer kernel launches
        precomp = model.precompute_sequence_features(frames[:, :T])
        # precomp['f4']:  [B, T, Cf, h, w]
        # precomp['g_s']: [B, T, 1, H, W]  (not used in Stage 2 but computed)

        # ── Collect boundary-frame attention weights (no grad needed) ─────
        boundary_weights = []
        with torch.no_grad():
            for t in range(T):
                if boundary_flags[:, t].any():
                    f_t = precomp['f4'][:, t]              # [B, Cf, h, w]
                    w   = model.channel_weight_extractor(f_t, gaze_coords[:, t])
                    boundary_weights.append(w)

        if len(boundary_weights) < 2:
            pbar.set_postfix({'att_loss': 0.0})
            continue

        # ── LSTM over boundary weights (trainable, with grad) ─────────────
        hidden    = None
        att_losses = []

        for i in range(len(boundary_weights) - 1):
            w_current    = boundary_weights[i]
            w_next_target = boundary_weights[i + 1].detach()

            w_pred, hidden = model.lstm_gated.forward_no_gate(w_current, hidden)

            att_loss = attention_criterion(w_pred, w_next_target)
            att_losses.append(att_loss)

        if att_losses:
            loss = torch.stack(att_losses).mean()
            n_boundary_steps  += len(att_losses)
            n_boundary_batches += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        pbar.set_postfix({'att_loss': loss.item() if att_losses else 0.0})

    n_batches_with_loss = max(n_boundary_batches, 1)
    return {
        'total':              total_loss / n_batches_with_loss,
        'attention':          total_loss / n_batches_with_loss,
        'n_boundary_steps':   n_boundary_steps,
        'n_boundary_batches': n_boundary_batches,
    }


def validate_stage2(model, loader, attention_criterion, device, config=None, epoch=None):
    """
    Validate Stage 2 (LSTM attention transition).

    Uses forward_no_gate throughout — fixation labels are not needed because
    the gate is bypassed, making metrics a clean measure of LSTM transition
    prediction independent of gating behaviour.
    """
    model.eval()

    log_predictions = config is not None and config.get('wandb', {}).get('log_predictions', False)
    num_samples = config.get('wandb', {}).get('num_prediction_samples', 4) if config else 4
    sample_frames = []
    sample_g_a    = []
    sample_gt_points = []

    total_loss = 0
    all_auc    = []
    all_nss    = []
    all_aae    = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating Stage 2"):
            frames         = batch['frames'].to(device)
            gaze_coords    = batch['gaze_coords'].to(device)
            boundary_flags = batch['boundary_flags'].to(device)
            seq_len        = batch['seq_len']
            T              = seq_len[0]

            precomp = model.precompute_sequence_features(frames[:, :T])

            hidden    = None
            att_losses = []

            for t in range(T - 1):
                f_t            = precomp['f4'][:, t + 1]
                f_t_minus_1    = precomp['f4'][:, t]
                gaze_t_minus_1 = gaze_coords[:, t]
                gaze_t         = gaze_coords[:, t + 1]
                is_boundary    = boundary_flags[:, t + 1]

                w_t_minus_1 = model.channel_weight_extractor(f_t_minus_1, gaze_t_minus_1)
                w_t_pred, hidden = model.lstm_gated.forward_no_gate(w_t_minus_1, hidden)

                if is_boundary.any():
                    w_t_target   = model.channel_weight_extractor(f_t, gaze_t)
                    boundary_mask = is_boundary.squeeze(-1)
                    att_losses.append(attention_criterion(
                        w_t_pred[boundary_mask],
                        w_t_target[boundary_mask]
                    ))

                    g_a_pred  = model.attention_weight_app(f_t, w_t_pred)
                    gt_heatmap = torch.zeros_like(g_a_pred)
                    for i in range(gaze_t.shape[0]):
                        gx = int(gaze_t[i, 0].clamp(0, 299))
                        gy = int(gaze_t[i, 1].clamp(0, 299))
                        gt_heatmap[i, 0, gy, gx] = 1.0

                    g_a_cpu = g_a_pred.cpu()
                    gt_cpu  = gt_heatmap.cpu()
                    all_auc.append(compute_auc(g_a_cpu, gt_cpu))
                    all_nss.append(compute_nss(g_a_cpu, gt_cpu))
                    all_aae.append(compute_aae(g_a_cpu, gt_cpu))

                    if log_predictions and len(sample_frames) < num_samples:
                        raw_frame_t = frames[:, t + 1]
                        for i in range(min(raw_frame_t.shape[0], num_samples - len(sample_frames))):
                            sample_frames.append(raw_frame_t[i].cpu())
                            sample_g_a.append(g_a_pred[i, 0].cpu())
                            sample_gt_points.append(gaze_t[i].cpu())

                hidden = (hidden[0].detach(), hidden[1].detach())

            loss = torch.stack(att_losses).mean() if att_losses else torch.tensor(0.0)
            total_loss += loss.item()

    n_batches = len(loader)
    auc = float(np.mean(all_auc)) if all_auc else 0.0
    nss = float(np.mean(all_nss)) if all_nss else 0.0
    aae = float(np.mean(all_aae)) if all_aae else 0.0
    print(f"\n  Stage 2 attention metrics (at boundaries):")
    print(f"    AUC: {auc:.4f} | NSS: {nss:.4f} | AAE: {aae:.2f}°")

    if log_predictions and config.get('wandb', {}).get('enabled', False) and len(sample_frames) > 0 and epoch is not None:
        log_stage2_predictions(sample_frames, sample_g_a, sample_gt_points, epoch)

    return {
        'loss':           total_loss / n_batches,
        'attention_loss': total_loss / n_batches,
        'auc':            auc,
        'nss':            nss,
        'aae':            aae,
    }


def log_stage2_predictions(frames, g_a_maps, gt_points, epoch):
    """Log Stage 2 attention transition predictions to W&B (2-panel: frame+GT, G_a)."""
    import matplotlib.pyplot as plt
    
    n = len(frames)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        img_np = frames[i].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
        axes[i, 0].imshow(img_np)
        gx, gy = gt_points[i][0].item(), gt_points[i][1].item()
        axes[i, 0].plot(gx, gy, 'r+', markersize=15, markeredgewidth=2)
        axes[i, 0].set_title("Frame + GT gaze")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(g_a_maps[i].numpy(), cmap='hot')
        axes[i, 1].set_title("G_a (attention transition)")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    wandb.log({f"stage2_predictions/epoch_{epoch}": wandb.Image(fig)})
    plt.close(fig)


# =============================================================================
# Stage 3: Training & Validation (GPU heatmap, AMP, visualization)
# =============================================================================

def train_stage3_epoch(model, loader, criterion, optimizer, device,
                       gaussian_kernel=None, use_amp=False, grad_accum_steps=1):
    """
    Train one epoch for Stage 3 (fusion) using SEQUENCES.

    Speedup vs original:
      • backbone + temporal_diff + saliency_decoder run ONCE per sequence as a
        single [B*T, C, H, W] batch (precompute_sequence_features).
      • The sequential loop only touches the cheap trainable path:
        channel_weight_extractor → lstm_gated → attention_weight_app → gated_fusion.
      • This eliminates ~90% of per-step FLOPs for Stage 3.

    Other optimisations preserved:
      • Mixed precision (use_amp)
      • Gradient accumulation (grad_accum_steps)
      • GPU heatmap generation (gaussian_kernel)
    """
    model.train()
    set_frozen_modules_eval(model, stage=3)
    total_loss = 0
    n_steps    = 0

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pbar = tqdm(loader, desc="Training Stage 3")
    for batch_idx, batch in enumerate(pbar):
        frames      = batch['frames'].to(device)       # [B, T, C, H, W]
        gaze_coords = batch['gaze_coords'].to(device)  # [B, T, 2]
        seq_len     = batch['seq_len']
        T           = seq_len[0]

        # ── Pre-compute frozen backbone + saliency for all T frames ───────
        # One [B*T, ...] backbone pass instead of T separate calls.
        precomp = model.precompute_sequence_features(frames[:, :T])
        # precomp['f4']:  [B, T, Cf, h, w]   — backbone features
        # precomp['g_s']: [B, T, 1, H, W]    — saliency maps (frozen)

        # ── Sequential LSTM loop — cheap trainable path only ───────────────
        hidden      = None
        batch_losses = []

        for t in range(1, T):
            f_t         = precomp['f4'][:, t]          # [B, Cf, h, w]
            f_t_minus_1 = precomp['f4'][:, t - 1]      # [B, Cf, h, w]
            g_s_t       = precomp['g_s'][:, t]         # [B, 1, H, W]
            gaze_t      = gaze_coords[:, t]             # [B, 2]
            gaze_t_minus_1 = gaze_coords[:, t - 1]     # [B, 2]

            hist_start   = max(0, t - FIXATION_IDT_WINDOW)
            gaze_history = [gaze_coords[:, i] for i in range(hist_start, t)]

            with torch.autocast('cuda', enabled=use_amp):
                # Fixation state (pure tensor ops, no backbone)
                fixation = model.compute_fixation_state(gaze_history, method='idt')

                # Trainable path — run LSTM in fp32 even under AMP. cuDNN LSTM
                # under autocast is known to produce inf grads when trainable.
                w_t_minus_1 = model.channel_weight_extractor(f_t_minus_1, gaze_t_minus_1)
                with torch.autocast('cuda', enabled=False):
                    w_t_fp32, hidden = model.lstm_gated(
                        w_t_minus_1.float(), fixation.float(), hidden
                    )
                w_t = w_t_fp32.to(w_t_minus_1.dtype)
                g_a_t       = model.attention_weight_app(f_t, w_t)
                g_t, gaze_pred = model.gated_fusion(g_s_t, g_a_t, return_logits=True)

                hidden = (hidden[0].detach(), hidden[1].detach())

                # GT heatmap — GPU path if kernel provided
                if gaussian_kernel is not None:
                    gt_heatmap = create_heatmap_gpu(gaze_t, gaussian_kernel, size=300)
                else:
                    batch_size = gaze_t.shape[0]
                    gt_heatmaps = []
                    for i in range(batch_size):
                        gx = gaze_t[i, 0].item() / SCALE_FACTOR_X
                        gy = gaze_t[i, 1].item() / SCALE_FACTOR_Y
                        hm = create_gaze_heatmap(gx, gy)
                        gt_heatmaps.append(torch.from_numpy(hm).float())
                    gt_heatmap = torch.stack(gt_heatmaps).unsqueeze(1).to(device)

                loss = criterion(g_t, gt_heatmap, gaze_t)

            batch_losses.append(loss)

        if batch_losses:
            total_batch_loss = torch.stack(batch_losses).mean() / grad_accum_steps

            scaler.scale(total_batch_loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += total_batch_loss.item() * grad_accum_steps
            n_steps    += 1

        pbar.set_postfix({'loss': total_batch_loss.item() * grad_accum_steps if batch_losses else 0.0})

    # Flush remaining accumulated gradients
    if n_steps > 0 and n_steps % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


def validate_stage3(model, loader, criterion, device, config=None, epoch=None,
                    gaussian_kernel=None):
    """
    Validate Stage 3 (full model with fusion) using SEQUENCES.
    Uses distance-based F1, pixel-level F1, and logs 6-panel visualizations.
    """
    model.eval()
    total_loss = 0
    n_steps    = 0
    all_auc    = []
    all_nss    = []
    all_aae    = []

    # For visualization
    log_predictions = config is not None and config.get('wandb', {}).get('log_predictions', False)
    num_samples = config.get('wandb', {}).get('num_prediction_samples', 4) if config else 4
    sample_frames = []
    sample_g_s = []
    sample_g_a = []
    sample_g_t = []
    sample_gt = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating Stage 3"):
            frames = batch['frames'].to(device)
            gaze_coords = batch['gaze_coords'].to(device)
            seq_len = batch['seq_len']
            
            hidden = None
            
            for t in range(1, seq_len[0]):
                frame_t = frames[:, t]
                frame_t_minus_1 = frames[:, t - 1]
                gaze_t = gaze_coords[:, t]
                gaze_t_minus_1 = gaze_coords[:, t - 1]
                
                hist_start = max(0, t - FIXATION_IDT_WINDOW)
                gaze_history = [gaze_coords[:, i] for i in range(hist_start, t)]

                # Request intermediates only when collecting viz samples
                need_intermediates = log_predictions and len(sample_frames) < num_samples
                
                # Get logits for loss computation
                result_logits = model(
                    frame_t, frame_t_minus_1, gaze_t_minus_1,
                    hidden=hidden, gaze_history=gaze_history,
                    fixation_method='idt',
                    return_logits=True,
                    return_intermediates=need_intermediates
                )
                
                if need_intermediates:
                    g_t_logits, gaze_pred, hidden, fixation, intermediates = result_logits
                else:
                    g_t_logits, gaze_pred, hidden, fixation = result_logits
                    intermediates = None
                
                # Probabilities for metrics
                g_t_prob = torch.sigmoid(g_t_logits) if model.gated_fusion.returns_logits else g_t_logits
                
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Generate GT heatmap
                if gaussian_kernel is not None:
                    gt_heatmap = create_heatmap_gpu(gaze_t, gaussian_kernel, size=300)
                else:
                    batch_size = gaze_t.shape[0]
                    gt_heatmaps = []
                    for i in range(batch_size):
                        gx = gaze_t[i, 0].item() / SCALE_FACTOR_X
                        gy = gaze_t[i, 1].item() / SCALE_FACTOR_Y
                        hm = create_gaze_heatmap(gx, gy)
                        gt_heatmaps.append(torch.from_numpy(hm).float())
                    gt_heatmap = torch.stack(gt_heatmaps).unsqueeze(1).to(device)
                
                loss = criterion(g_t_logits, gt_heatmap, gaze_t)
                total_loss += loss.item()
                n_steps += 1
                
                # Metrics (move to CPU once, reuse for all three)
                g_t_cpu = g_t_prob.cpu()
                gt_cpu  = gt_heatmap.cpu()
                all_auc.append(compute_auc(g_t_cpu, gt_cpu))
                all_nss.append(compute_nss(g_t_cpu, gt_cpu))
                all_aae.append(compute_aae(g_t_cpu, gt_cpu))
                
                # Collect visualization samples
                if intermediates is not None and len(sample_frames) < num_samples:
                    for i in range(min(frame_t.shape[0], num_samples - len(sample_frames))):
                        sample_frames.append(frame_t[i].cpu())
                        g_s_viz = intermediates['g_s'][i, 0]
                        if hasattr(model.saliency_decoder, 'return_logits') and model.saliency_decoder.return_logits:
                            g_s_viz = torch.sigmoid(g_s_viz)
                        sample_g_s.append(g_s_viz.cpu())
                        sample_g_a.append(intermediates['g_a'][i, 0].cpu())
                        sample_g_t.append(intermediates['g_t'][i, 0].cpu())
                        sample_gt.append(gt_heatmap[i, 0].cpu())
    
    # Log 5-panel visualization
    if log_predictions and config.get('wandb', {}).get('enabled', False) and len(sample_frames) > 0 and epoch is not None:
        log_stage3_predictions(sample_frames, sample_g_s, sample_g_a, sample_g_t, sample_gt, epoch)
    
    return {
        'loss': total_loss / max(n_steps, 1),
        'auc':  float(np.mean(all_auc)) if all_auc else 0.0,
        'nss':  float(np.mean(all_nss)) if all_nss else 0.0,
        'aae':  float(np.mean(all_aae)) if all_aae else 0.0,
    }


def log_stage3_predictions(frames, g_s_maps, g_a_maps, g_t_maps, gt_maps, epoch):
    """
    Log Stage 3 six-panel predictions to W&B:
    Frame, G_s, G_a, G_t, GT, Gaze points overlay.
    
    Last column shows the input frame with predicted gaze point (green +)
    and GT gaze point (red +), plus a small Gaussian at the predicted point
    to match the GT heatmap visualization style.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    
    n = len(frames)
    fig, axes = plt.subplots(n, 6, figsize=(24, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = ["Input frame", "G_s (saliency)", "G_a (attention)", 
                  "G_t (fused)", "Ground truth", "Gaze points"]
    
    for i in range(n):
        img_np = frames[i].permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
        
        # Col 0: Input frame
        axes[i, 0].imshow(img_np)
        
        # Col 1: G_s saliency
        axes[i, 1].imshow(g_s_maps[i].numpy(), cmap='hot')
        
        # Col 2: G_a attention (normalize to [0,1] for better visualization)
        g_a_np = g_a_maps[i].numpy()
        g_a_min, g_a_max = g_a_np.min(), g_a_np.max()
        if g_a_max > g_a_min:
            g_a_np = (g_a_np - g_a_min) / (g_a_max - g_a_min)
        axes[i, 2].imshow(g_a_np, cmap='hot')
        
        # Col 3: G_t fused
        axes[i, 3].imshow(g_t_maps[i].numpy(), cmap='hot')
        
        # Col 4: Ground truth
        axes[i, 4].imshow(gt_maps[i].numpy(), cmap='hot')
        
        # Col 5: Gaze points — frame with pred and GT points overlaid
        g_t_np = g_t_maps[i].numpy()
        gt_np = gt_maps[i].numpy()
        
        # Predicted gaze: argmax of G_t
        pred_idx = np.unravel_index(g_t_np.argmax(), g_t_np.shape)
        pred_y, pred_x = pred_idx[0], pred_idx[1]
        
        # GT gaze: argmax of GT heatmap
        gt_idx = np.unravel_index(gt_np.argmax(), gt_np.shape)
        gt_y, gt_x = gt_idx[0], gt_idx[1]
        
        # Create predicted gaze heatmap with same sigma as GT
        pred_hm = np.zeros_like(g_t_np)
        pred_hm[pred_y, pred_x] = 1.0
        pred_hm = gaussian_filter(pred_hm, sigma=16.4)
        if pred_hm.max() > 0:
            pred_hm /= pred_hm.max()
        
        # Show frame with heatmap overlay and gaze points
        axes[i, 5].imshow(img_np)
        axes[i, 5].imshow(pred_hm, cmap='hot', alpha=0.4)
        axes[i, 5].plot(pred_x, pred_y, 'g+', markersize=18, markeredgewidth=3, label='Pred')
        axes[i, 5].plot(gt_x, gt_y, 'r+', markersize=18, markeredgewidth=3, label='GT')
        if i == 0:
            axes[i, 5].legend(loc='upper right', fontsize=8)
        
        for j in range(6):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j])
    
    plt.tight_layout()
    wandb.log({f"stage3_predictions/epoch_{epoch}": wandb.Image(fig)})
    plt.close(fig)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model, optimizer, stage, epoch, metrics, config, filename,
                    scheduler=None):
    """Save checkpoint."""
    checkpoint = {
        'stage': stage,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'torch_rng_state': torch.random.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
    }
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    
    path = os.path.join(config['checkpoint_dir'], filename)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint, skipping mismatched layers.

    Handles all combinations of compiled / uncompiled checkpoints and models:
      - compiled   → compiled   (same keys, direct match)
      - compiled   → uncompiled (ckpt has _orig_mod., model doesn't)
      - uncompiled → compiled   (model has _orig_mod., ckpt doesn't)
      - uncompiled → uncompiled (same keys, direct match)

    Strategy: strip _orig_mod. and module. from BOTH sides when building the
    match index, then write into the model using its own (unstripped) keys.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state     = model.state_dict()
    pretrained_state = checkpoint['model_state_dict']

    def _strip(k):
        return k.replace('_orig_mod.', '').replace('module.', '')

    # Build a lookup: stripped_key → model_actual_key
    stripped_to_model = {_strip(k): k for k in model_state}

    # For each checkpoint param, find the matching model key by stripped name
    filtered_state = {}
    for ck_key, ck_val in pretrained_state.items():
        model_key = stripped_to_model.get(_strip(ck_key))
        if model_key is not None and ck_val.shape == model_state[model_key].shape:
            filtered_state[model_key] = ck_val

    loaded  = len(filtered_state)
    skipped = len(pretrained_state) - loaded
    if loaded == 0:
        ck_sample = next(iter(pretrained_state.keys()))
        m_sample  = next(iter(model_state.keys()))
        print(f"WARNING: load_checkpoint matched 0/{len(pretrained_state)} keys.")
        print(f"  sample ckpt  key: {ck_sample}")
        print(f"  sample model key: {m_sample}")
    elif skipped > 0:
        print(f"  load_checkpoint: {skipped} keys skipped (size mismatch or not in model)")

    model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded checkpoint: {loaded} params loaded, {skipped} skipped")
    return checkpoint


def verify_checkpoint(model, checkpoint_path, device, val_loader=None, criterion=None):
    """
    Load checkpoint and optionally run a quick validation sanity check.
    
    Returns loaded checkpoint dict.
    """
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    print(f"  Stage: {checkpoint.get('stage', '?')}, Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Saved metrics: {checkpoint.get('metrics', {})}")
    
    if val_loader is not None and criterion is not None:
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            if 'frame_t' in batch:
                frame_t = batch['frame_t'].to(device)
                frame_t_minus_1 = batch['frame_t_minus_1'].to(device)
                heatmap_t = batch['heatmap_t'].to(device)
                g_s = model.forward_saliency_only(frame_t, frame_t_minus_1)
                g_s_prob = torch.sigmoid(g_s) if hasattr(model.saliency_decoder, 'return_logits') and model.saliency_decoder.return_logits else g_s
                gt = heatmap_t.unsqueeze(1)
                auc = compute_auc(g_s_prob, gt)
                print(f"  Quick AUC check (1 batch): {auc:.4f}")
    
    return checkpoint


# =============================================================================
# Main Training Orchestrator
# =============================================================================

def train_stage(model, stage, config, device, train_loader, val_loader, 
                criterion, optimizer, stage_config):
    """Train a single stage with early stopping and comprehensive logging."""
    best_val_loss = float('inf')
    best_val_auc = -1.0
    patience_counter = 0
    best_metrics = None
    global_step = 0
    
    log_predictions = config['wandb'].get('log_predictions', False)
    
    # Pre-compute Gaussian kernel for GPU heatmap generation (Stage 3)
    gaussian_kernel = None
    if stage == 3:
        gaussian_kernel = make_gaussian_kernel(GAUSSIAN_SIGMA).to(device)
        print(f"GPU heatmap kernel ready (sigma={GAUSSIAN_SIGMA:.1f})")
    
    for epoch in range(stage_config['epochs']):
        print(f"\n=== Stage {stage} | Epoch {epoch+1}/{stage_config['epochs']} ===")
        
        # Training
        if stage == 1:
            train_loss, global_step = train_stage1_epoch(
                model, train_loader, criterion, optimizer, device, config, epoch + 1, global_step
            )
            train_metrics = {'loss': train_loss}
        elif stage == 2:
            train_metrics = train_stage2_epoch(
                model, train_loader, 
                AttentionTransitionLoss(), 
                optimizer, device
            )
            train_loss = train_metrics['total']
        else:
            train_loss = train_stage3_epoch(
                model, train_loader, criterion, optimizer, device,
                gaussian_kernel=gaussian_kernel,
                use_amp=config.get('training', {}).get('use_amp', False),
                grad_accum_steps=config.get('training', {}).get('grad_accum_steps', 1),
            )
            train_metrics = {'loss': train_loss}
        
        # Validation
        if stage == 1:
            val_metrics = validate_stage1(
                model, val_loader, criterion, device, config, epoch + 1,
                log_predictions=log_predictions
            )
        elif stage == 2:
            val_metrics = validate_stage2(
                model, val_loader,
                AttentionTransitionLoss(), 
                device, config=config, epoch=epoch + 1
            )
        else:
            val_metrics = validate_stage3(
                model, val_loader, criterion, device,
                config=config, epoch=epoch + 1,
                gaussian_kernel=gaussian_kernel
            )
        
        val_loss = val_metrics['loss']
        
        # Logging
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if 'auc' in val_metrics:
            aae_val = val_metrics.get('aae', 0.0)
            print(f"Val AUC: {val_metrics['auc']:.4f} | Val AAE: {aae_val:.4f}")
        if 'distance_f1' in val_metrics:
            print(f"Val Hit Rate (2σ): {val_metrics['distance_f1']:.1f} | Val Pixel F1: {val_metrics.get('pixel_f1', 0):.1f} (P={val_metrics.get('pixel_precision', 0):.1f} R={val_metrics.get('pixel_recall', 0):.1f})")
        
        # W&B logging
        if config['wandb']['enabled']:
            log_dict = {
                f'stage{stage}/train_loss': train_loss,
                f'stage{stage}/val_loss': val_loss,
                f'stage{stage}/epoch': epoch + 1
            }
            for key, value in val_metrics.items():
                log_dict[f'stage{stage}/val_{key}'] = value
            wandb.log(log_dict)
            
            if stage == 1 and hasattr(model.saliency_decoder, 'center_bias'):
                log_center_bias_visualization(model, epoch + 1)
        
        # Early stopping: AUC when available (all stages), loss as fallback
        if 'auc' in val_metrics:
            improved = val_metrics['auc'] > best_val_auc
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_val_auc = val_metrics.get('auc', best_val_auc)
            best_metrics = val_metrics.copy()
            patience_counter = 0

            save_checkpoint(
                model, optimizer, stage, epoch + 1, val_metrics, config,
                f'checkpoint_stage{stage}_best.pt'
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{stage_config['patience']}")

            if patience_counter >= stage_config['patience']:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    save_checkpoint(
        model, optimizer, stage, epoch + 1, val_metrics, config,
        f'checkpoint_stage{stage}_final.pt'
    )
    
    return best_metrics


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train GazeLite')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage to start from (1, 2, or 3)')
    parser.add_argument('--no-skip-connections', action='store_true',
                        help='Disable skip connections in saliency decoder')
    parser.add_argument('--no-center-bias', action='store_true',
                        help='Disable learnable center bias prior')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    
    # Setup
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Architecture options
    use_skip_connections = not args.no_skip_connections
    use_center_bias = not args.no_center_bias
    
    print(f"Architecture options:")
    print(f"  Skip connections: {use_skip_connections}")
    print(f"  Center bias: {use_center_bias}")
    
    
    #Center Bias
    center_bias_sigma = config.get('stage1', {}).get('center_bias_sigma', 0.4)

    #Fusion mode
    fusion_mode = config.get('stage3', {}).get('fusion_mode', 'huang')

    
    # Create model
    model = GazeLite(
        pretrained_backbone=True,
        use_skip_connections=use_skip_connections,
        use_center_bias=use_center_bias,
        center_bias_sigma=center_bias_sigma,
        fusion_mode=fusion_mode
    )
    model = model.to(device)

    # torch.compile: fuses CUDA kernels for ~20–40% free speedup on PyTorch 2+.
    # Set torch_compile: true in config to enable.
    #
    # NOTE: we compile individual submodules, NOT the outer model.  All three
    # training stages bypass model.forward() and call submodules directly, so
    # torch.compile(model) would have zero effect.  Compiling each submodule
    # means every call-site (Stage 1 forward_saliency_only, Stage 2/3 loops,
    # and precompute_sequence_features) benefits automatically.
    #
    # Each submodule triggers its own one-time compilation on first call (~seconds).
    if config.get('torch_compile', False):
        print("Compiling model submodules with torch.compile ...")
        model.backbone              = torch.compile(model.backbone)
        model.temporal_diff         = torch.compile(model.temporal_diff)
        model.saliency_decoder      = torch.compile(model.saliency_decoder)
        model.channel_weight_extractor = torch.compile(model.channel_weight_extractor)
        model.lstm_gated            = torch.compile(model.lstm_gated)
        model.attention_weight_app  = torch.compile(model.attention_weight_app)
        model.gated_fusion          = torch.compile(model.gated_fusion)
        print("  Compiled: backbone, temporal_diff, saliency_decoder, "
              "channel_weight_extractor, lstm_gated, attention_weight_app, gated_fusion")

    # Initialize W&B (after model creation for wandb.watch)
    wandb_enabled = setup_wandb(config, model)
    
    # Load checkpoint if resuming
    start_stage = args.stage
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
    
    # Log model info
    total_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    if wandb_enabled:
        wandb.log({
            'model/parameters': total_params,
            'model/size_mb': model_size,
            'model/skip_connections': use_skip_connections,
            'model/center_bias': use_center_bias
        })
    
    # ==================== STAGE 1 ====================
    if start_stage <= 1:
        print("\n" + "="*60)
        print("STAGE 1: Training Saliency Decoder")
        print("="*60)
        
        setup_model_for_stage(model, stage=1, config=config)

        gt_sigma = config['stage1'].get('gt_sigma', None)
        train_loader, val_loader = create_dataloaders(
            config['data_root'],
            config['splits_json'],
            option=config.get('option', 'b'),
            batch_size=config['stage1']['batch_size'],
            num_workers=config.get('num_workers', 8),
            sample_fraction=config.get('sample_fraction', 1.0),
            val_sample_fraction=config.get('val_sample_fraction', None),
            gt_sigma=gt_sigma,
            cache_dir=config.get('cache_dir', None),
            frames_dir=config.get('frames_dir', None),
        )
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['stage1']['lr']
        )

        loss_function = config['stage1'].get('loss_function', 'bce')
        if loss_function == 'bce':
            criterion = SaliencyLoss()
            model.saliency_decoder.return_logits = False
        elif loss_function == 'bce_logits':
            criterion = SaliencyLogitsLoss()
            model.saliency_decoder.return_logits = True
        elif loss_function == 'kld':
            criterion = SaliencyKLDLoss()
            model.saliency_decoder.return_logits = True
        else:
            raise ValueError(f"Unknown loss function: '{loss_function}'. Options: 'bce', 'bce_logits', 'kld'")
        print(f"  Loss function: {loss_function}")
        
        
        stage1_metrics = train_stage(
            model, stage=1, config=config, device=device,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
            stage_config=config['stage1']
        )
        
        print(f"Stage 1 complete. Best AUC: {stage1_metrics.get('auc', 'N/A')}")
        
        # Exit if stage1_only is set
        if config.get('stage1_only', False):
            print("\nstage1_only=True, exiting after Stage 1.")
            if wandb_enabled:
                wandb.finish()
            print("Done!")
            sys.exit(0)
    
    # ==================== STAGE 2 ====================
    if start_stage <= 2:
        print("\n" + "="*60)
        print("STAGE 2: Training Attention Transition (LSTM)")
        print("="*60)
        
        setup_model_for_stage(model, stage=2)
        
        train_loader = create_sequence_dataloader(
            config['data_root'],
            config['splits_json'],
            split='train',
            option=config.get('option', 'b'),
            batch_size=config['stage2']['batch_size'],
            num_workers=config.get('num_workers', 8),
            sample_fraction=config.get('sample_fraction', 1.0),
            max_seq_len=config.get('bptt_window', 64),
            cache_dir=config.get('cache_dir', None),
            frames_dir=config.get('frames_dir', None),
        )
        val_loader = create_sequence_dataloader(
            config['data_root'],
            config['splits_json'],
            split='val',
            option=config.get('option', 'b'),
            batch_size=config['stage2']['batch_size'],
            num_workers=config.get('num_workers', 8),
            sample_fraction=config.get('val_sample_fraction') or config.get('sample_fraction', 1.0),
            max_seq_len=config.get('bptt_window', 64),
            cache_dir=config.get('cache_dir', None),
            frames_dir=config.get('frames_dir', None),
        )
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['stage2']['lr']
        )
        
        criterion = None
        
        stage2_metrics = train_stage(
            model, stage=2, config=config, device=device,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
            stage_config=config['stage2']
        )
        
        print(f"Stage 2 complete.")

         # Exit if stage2_only is set
        if config.get('stage2_only', False):
            print("\nstage2_only=True, exiting after Stage 2.")
            if wandb_enabled:
                wandb.finish()
            print("Done!")
            sys.exit(0)
    
    # ==================== STAGE 3 ====================
    if start_stage <= 3:
        print("\n" + "="*60)
        print("STAGE 3: Training Fusion (Sequence)")
        print("="*60)

        setup_model_for_stage(model, stage=3, config=config)
        
        train_loader = create_sequence_dataloader(
            config['data_root'],
            config['splits_json'],
            split='train',
            option=config.get('option', 'b'),
            batch_size=config['stage3'].get('batch_size', 1),
            num_workers=config.get('num_workers', 8),
            sample_fraction=config.get('sample_fraction', 1.0),
            max_seq_len=config.get('bptt_window', 64),
            cache_dir=config.get('cache_dir', None),
            frames_dir=config.get('frames_dir', None),
        )
        val_loader = create_sequence_dataloader(
            config['data_root'],
            config['splits_json'],
            split='val',
            option=config.get('option', 'b'),
            batch_size=config['stage3'].get('batch_size', 1),
            num_workers=config.get('num_workers', 8),
            sample_fraction=config.get('val_sample_fraction') or config.get('sample_fraction', 1.0),
            max_seq_len=config.get('bptt_window', 64),
            cache_dir=config.get('cache_dir', None),
            frames_dir=config.get('frames_dir', None),
        )
        
        # Build optimizer — separate param groups if co-training attention path,
        # so we can give the (small, randomly init) fusion module a higher lr
        # than the (already-trained, easy to destabilize) attention path.
        cotrain_attention = config.get('stage3', {}).get('cotrain_attention', False)
        fusion_lr = config['stage3']['lr']
        if cotrain_attention:
            attention_lr = config['stage3'].get('attention_lr', fusion_lr * 0.1)
            attention_params = []
            for m in (model.channel_weight_extractor, model.lstm_gated,
                      model.attention_weight_app):
                attention_params += [p for p in m.parameters() if p.requires_grad]
            fusion_params = [p for p in model.gated_fusion.parameters()
                             if p.requires_grad]
            optimizer = optim.Adam([
                {'params': fusion_params,    'lr': fusion_lr},
                {'params': attention_params, 'lr': attention_lr},
            ])
            print(f"  Stage 3 lrs: fusion={fusion_lr}, attention={attention_lr}")
        else:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=fusion_lr
            )

        criterion = FusionLoss()
        
        stage3_metrics = train_stage(
            model, stage=3, config=config, device=device,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
            stage_config=config['stage3']
        )
        
        print(f"Stage 3 complete. Final AUC: {stage3_metrics.get('auc', 'N/A')}")
    
    # ==================== FINAL ====================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    save_checkpoint(
        model, optimizer, stage=3, epoch=0, 
        metrics=stage3_metrics if start_stage <= 3 else {}, 
        config=config,
        filename='checkpoint_final.pt'
    )
    
    if wandb_enabled:
        wandb.finish()
    
    print("Done!")


if __name__ == "__main__":
    main()