"""
Evaluation metrics for GazeLite.
Includes gaze prediction metrics, fixation metrics, and efficiency utilities.

All saliency/gaze metrics follow Huang et al. (2018) methodology:
    - Predicted gaze point: center-of-mass of predicted heatmap (weighted centroid)
    - Ground truth gaze point: argmax of ground truth heatmap (peak location)
    - AAE: 3D ray projection with pinhole camera model (60° FOV)
    - AUC: Judd-style saliency AUC

Distance-based F1 (our addition):
    - Both pred and GT: argmax of heatmap → point coordinate
    - Hit/miss based on Euclidean distance at multiple radii
    - Avoids threshold ambiguity between sigmoid and softmax outputs
    - Directly interpretable: "X% of frames predicted within Y pixels"

Reference implementations verified against:
    - Huang et al.: github.com/hyf015/egocentric-gaze-prediction (utils.py)
    - Lai et al.: github.com/BolinLai/GLC (slowfast/utils/metrics.py)
"""

import torch
import math
import numpy as np
from scipy import ndimage


# =============================================================================
# Gaze Point Extraction Utilities
# =============================================================================

def extract_gaze_points(pred_heatmap, target_heatmap, pred_method='com', gt_method='argmax'):
    """
    Extract gaze coordinates from predicted and ground truth heatmaps.
    
    Following Huang et al. (2018) defaults:
        - Predicted gaze: center-of-mass (weighted centroid, robust to noise)
        - GT gaze: argmax (peak of the Gaussian placed at true gaze location)
    
    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        pred_method: 'com' (center-of-mass) or 'argmax'
        gt_method: 'com' or 'argmax'
        
    Returns:
        pred_points: np.array [batch, 2] as (x, y) in pixel coordinates
        gt_points: np.array [batch, 2] as (x, y) in pixel coordinates
    """
    pred = pred_heatmap.detach().cpu().numpy()
    target = target_heatmap.detach().cpu().numpy()
    
    batch_size = pred.shape[0]
    pred_points = np.zeros((batch_size, 2))
    gt_points = np.zeros((batch_size, 2))
    
    for b in range(batch_size):
        pred_points[b] = _extract_point(pred[b, 0], pred_method)
        gt_points[b] = _extract_point(target[b, 0], gt_method)
    
    return pred_points, gt_points


def _extract_point(heatmap_2d, method='com'):
    """Extract a single (x, y) point from a 2D heatmap."""
    if method == 'com':
        com = ndimage.measurements.center_of_mass(heatmap_2d)
        if np.isnan(com[0]) or np.isnan(com[1]):
            # Fallback to center if heatmap is all zeros
            return np.array([heatmap_2d.shape[1] / 2, heatmap_2d.shape[0] / 2])
        return np.array([com[1], com[0]])  # (x, y) from (row, col)
    elif method == 'argmax':
        idx = np.unravel_index(heatmap_2d.argmax(), heatmap_2d.shape)
        return np.array([idx[1], idx[0]])  # (x, y) from (row, col)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


# =============================================================================
# Distance-Based F1 / Precision / Recall (NEW — replaces threshold-based F1)
# =============================================================================

def compute_distance_f1(pred_heatmap, target_heatmap, radii=None, sigma=None):
    """
    Compute frame-level F1, Precision, Recall using point distance.
    
    Unlike threshold-based approaches (Lai et al. adaptive_f1), this method:
    1. Extracts point predictions via argmax from both pred and GT heatmaps
    2. Computes Euclidean distance between predicted and GT points
    3. Classifies each frame as hit (distance < radius) or miss
    4. Computes F1 = 2*P*R / (P+R) at each radius
    
    This avoids the sigmoid-vs-softmax threshold incompatibility entirely.
    
    Default radii correspond to 1σ, 2σ, 3σ of the GT Gaussian (σ ≈ 16.4px):
        - 1σ (~16px): strict, only very accurate predictions
        - 2σ (~33px): moderate, within the central region of GT Gaussian
        - 3σ (~49px): lenient, within the visible extent of GT Gaussian
    
    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        radii: List of distance thresholds in pixels. Default: [16, 33, 49]
        sigma: If provided, radii = [1*sigma, 2*sigma, 3*sigma]
        
    Returns:
        Dictionary with per-radius results:
        {
            'distance_f1_{r}': F1 at radius r (0-100 scale),
            'distance_precision_{r}': Precision at radius r,
            'distance_recall_{r}': Recall at radius r,
            'mean_distance': mean Euclidean distance across batch,
            'hit_rate_{r}': fraction of frames within radius r,
        }
        Also includes 'distance_f1' as the F1 at the middle (2σ) radius for
        use as a single summary metric.
    """
    if sigma is not None:
        radii = [1 * sigma, 2 * sigma, 3 * sigma]
    elif radii is None:
        radii = [16, 33, 49]
    
    # Use argmax for both pred and GT (consistent point extraction)
    pred_points, gt_points = extract_gaze_points(
        pred_heatmap, target_heatmap,
        pred_method='argmax', gt_method='argmax'
    )
    
    # Euclidean distances
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))
    
    results = {
        'mean_distance': float(np.mean(distances)),
    }
    
    for i, r in enumerate(radii):
        hits = distances < r
        n_hits = hits.sum()
        n_total = len(distances)
        
        # In single-point-per-frame formulation:
        # Each frame has exactly one prediction and one GT point.
        # Hit → TP=1, FP=0, FN=0
        # Miss → TP=0, FP=1, FN=1
        # So over N frames: TP = n_hits, FP = n_total - n_hits, FN = n_total - n_hits
        tp = n_hits
        fp = n_total - n_hits
        fn = n_total - n_hits
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        hit_rate = float(n_hits / n_total) if n_total > 0 else 0.0
        
        r_int = int(round(r))
        results[f'distance_f1_{r_int}'] = f1
        results[f'distance_precision_{r_int}'] = precision
        results[f'distance_recall_{r_int}'] = recall
        results[f'hit_rate_{r_int}'] = hit_rate
    
    # Summary F1: use middle radius (2σ) as the single number
    mid_r = int(round(radii[len(radii) // 2]))
    results['distance_f1'] = results[f'distance_f1_{mid_r}']
    results['distance_precision'] = results[f'distance_precision_{mid_r}']
    results['distance_recall'] = results[f'distance_recall_{mid_r}']
    
    return results


# =============================================================================
# Pixel-Level F1 / Precision / Recall (Lai et al. IJCV 2024 compatible)
# =============================================================================

def compute_pixel_f1(pred_heatmap, target_heatmap, num_thresholds=11, gt_threshold=0.001):
    """
    Compute pixel-level F1, Precision, Recall following Lai et al. (IJCV 2024).
    
    Method:
        1. Binarize GT heatmap: pixels > gt_threshold → salient region
        2. Sweep thresholds on prediction to create binary salient region
        3. Compute pixel-level TP, FP, FN per frame
        4. Average precision and recall across frames, compute F1
        5. Pick threshold giving best F1
    
    Lai's original sweeps absolute thresholds 0.0-0.02 on softmax output.
    For sigmoid output, we sweep relative thresholds (fraction of peak)
    from 0.05 to 0.95, which adapts to the output distribution.
    
    This gives comparable precision/recall semantics to Lai's Table 2:
        - Precision: what fraction of your predicted salient region is correct
        - Recall: what fraction of the GT salient region did you cover
    
    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W] (sigmoid probabilities)
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        num_thresholds: Number of thresholds to sweep
        gt_threshold: Threshold for binarizing GT (Lai uses 0.001)
        
    Returns:
        Dictionary with pixel_f1, pixel_precision, pixel_recall, best_threshold
    """
    pred = pred_heatmap.detach()
    target = target_heatmap.detach()
    batch_size = pred.shape[0]
    
    # Binary GT: salient region
    binary_gt = (target.squeeze(1) > gt_threshold).float()  # [B, H, W]
    
    # Sweep relative thresholds (fraction of per-sample peak)
    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    
    best_f1 = -1
    best_results = None
    
    for t in thresholds:
        # Per-sample relative threshold
        pred_squeezed = pred.squeeze(1)  # [B, H, W]
        peak_vals = pred_squeezed.amax(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        binary_pred = (pred_squeezed > t * peak_vals).float()  # [B, H, W]
        
        # Pixel-level TP, FP, FN per frame
        tp = (binary_pred * binary_gt).sum(dim=(1, 2))          # [B]
        fg_pred = binary_pred.sum(dim=(1, 2))                    # [B]
        fg_gt = binary_gt.sum(dim=(1, 2))                        # [B]
        
        # Average precision and recall across frames
        precision_per_frame = tp / (fg_pred + 1e-6)
        recall_per_frame = tp / (fg_gt + 1e-6)
        
        mean_precision = precision_per_frame.mean().item() * 100
        mean_recall = recall_per_frame.mean().item() * 100
        
        f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_results = {
                'pixel_f1': f1,
                'pixel_precision': mean_precision,
                'pixel_recall': mean_recall,
                'pixel_f1_threshold': float(t),
            }
    
    return best_results


# =============================================================================
# Core Gaze Prediction Metrics
# =============================================================================

def compute_nss(pred_heatmap, target_heatmap):
    """
    Compute Normalized Scanpath Saliency (NSS).

    Unlike AUC/AAE which collapse the prediction to a CoM point, NSS uses
    the full predicted heatmap distribution:
        1. Z-score normalize the predicted heatmap (zero mean, unit std)
        2. Read off the value at the GT fixation location (argmax of target)
        3. Average across batch

    NSS > 0: prediction is above-average saliency at the fixation location
    NSS = 1.0: fixation falls exactly 1 std dev above mean saliency
    Higher is better. Well-performing models typically score 1.5–3.0.

    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]

    Returns:
        Mean NSS across batch
    """
    pred = pred_heatmap.detach().cpu().numpy()
    target = target_heatmap.detach().cpu().numpy()

    nss_scores = []

    for b in range(pred.shape[0]):
        saliency = pred[b, 0]  # [H, W]

        # Z-score normalize
        mu = saliency.mean()
        sigma = saliency.std()
        if sigma < 1e-8:
            nss_scores.append(0.0)
            continue
        saliency_norm = (saliency - mu) / sigma

        # GT fixation: argmax of target heatmap (consistent with AUC/AAE)
        gt_idx = np.unravel_index(target[b, 0].argmax(), target[b, 0].shape)
        nss_scores.append(float(saliency_norm[gt_idx[0], gt_idx[1]]))

    return float(np.mean(nss_scores))


def compute_auc(pred_heatmap, target_heatmap, gaussian_sigma=18.75):
    """
    Compute Judd-style AUC (Area Under the Curve) for saliency evaluation.
    
    Method (following Huang et al., 2018):
        1. Extract predicted gaze point as center-of-mass of predicted heatmap
        2. Place a Gaussian at the predicted point on a blank canvas
        3. Look up the saliency value at the ground truth gaze location
        4. AUC = 1 - (fraction of pixels with saliency > saliency_at_gt)
    
    Verified against:
        - Huang: sigma=14 at 224×224, d_auc=14/224=0.0625
        - Lai (GLC): sigma=3.2 at 64×64, d_auc=3.2/64=0.05
        - Ours: sigma=18.75 at 300×300, d_auc=18.75/300=0.0625 (matches Huang)
    
    Note: Lai's sigma ratio is slightly tighter than Huang's. When comparing
    AUC with Lai's numbers, this difference may cause minor discrepancies.
    For Huang comparison, our scaling is exact.
    
    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        gaussian_sigma: Sigma for Gaussian placed at predicted point.
        
    Returns:
        Mean Judd AUC across batch
    """
    pred = pred_heatmap.detach().cpu().numpy()
    target = target_heatmap.detach().cpu().numpy()
    
    batch_size = pred.shape[0]
    H, W = pred.shape[2], pred.shape[3]
    aucs = []
    
    for b in range(batch_size):
        pred_map = pred[b, 0]
        target_map = target[b, 0]
        
        # Predicted gaze: center-of-mass (matches Huang & Lai)
        pred_point = ndimage.measurements.center_of_mass(pred_map)
        if np.isnan(pred_point[0]) or np.isnan(pred_point[1]):
            # Fallback: center of image (matches Lai's NaN handling)
            pred_y, pred_x = H // 2, W // 2
        else:
            pred_y = int(np.clip(round(pred_point[0]), 0, H - 1))
            pred_x = int(np.clip(round(pred_point[1]), 0, W - 1))
        
        # GT gaze: argmax (matches Huang & Lai)
        gt_idx = np.unravel_index(target_map.argmax(), target_map.shape)
        gt_y, gt_x = gt_idx[0], gt_idx[1]
        
        # Place Gaussian at predicted point (Judd method)
        z = np.zeros((H, W))
        z[pred_y, pred_x] = 1
        z = ndimage.filters.gaussian_filter(z, gaussian_sigma)
        z = z - np.min(z)
        if np.max(z) > 0:
            z = z / np.max(z)
        
        # AUC = 1 - fraction of pixels more salient than GT location
        saliency_at_gt = z[gt_y, gt_x]
        fp_fraction = float((z > saliency_at_gt).sum()) / (H * W)
        auc = 1.0 - fp_fraction
        aucs.append(auc)
    
    return np.mean(aucs)


def compute_aae(pred_heatmap, target_heatmap, image_size=300, fov_degrees=None):
    """
    Compute Average Angular Error (AAE) following Huang et al. (2018).

    Projects predicted and GT gaze points onto 3D rays using a pinhole camera
    model, then computes the angle between those rays.

    Camera model
    ------------
    focal_length = (image_size / 2) / tan(hfov / 2)

    where hfov is the horizontal half-FOV of the camera used to record the data.

    EGTEA Gaze+ (default, fov_degrees=None):
        Uses 60° full-FOV (half-FOV = 30° = π/6), matching Huang et al. (2018)
        and Lai et al. — results are directly comparable to their published numbers.
        d = (size/2) / tan(π/6)

    Ego-Exo-4D / Project Aria RGB (fov_degrees=110):
        The Aria RGB camera has ~110° diagonal FOV.  For the square 1408×1408
        perspective-projected frame the horizontal FOV is ~78°, giving:
        d = (size/2) / tan(39° in radians)
        Using the wrong (EGTEA) FOV here would over-estimate focal length and
        systematically report lower AAE than the true angular error.

        Pass fov_degrees=78 (horizontal) when evaluating on Ego-Exo-4D.
        Example:
            compute_aae(pred, gt, image_size=300, fov_degrees=78)

    Gaze extraction: argmax (pred), argmax (GT).
    Note: argmax is used for pred (rather than CoM) to avoid center bias
    dragging the CoM toward image center for peripheral fixations.

    Args:
        pred_heatmap:   Predicted heatmap  [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        image_size:     Image size in pixels (default 300)
        fov_degrees:    Full horizontal FOV of the recording camera in degrees.
                        None → use 60° (EGTEA / Huang et al. default).
                        Set to 78 for Ego-Exo-4D / Aria RGB.

    Returns:
        Mean angular error in degrees
    """
    pred   = pred_heatmap.detach().cpu().numpy()
    target = target_heatmap.detach().cpu().numpy()

    batch_size = pred.shape[0]
    center = image_size / 2.0

    # Focal length from camera FOV
    if fov_degrees is None:
        # EGTEA default: 60° full-FOV → matches Huang / Lai implementations
        d = center / math.tan(math.pi / 6)
    else:
        half_fov_rad = math.radians(fov_degrees / 2.0)
        d = center / math.tan(half_fov_rad)

    aae_list = []

    for b in range(batch_size):
        # Predicted gaze: argmax (returns row, col = y, x)
        # Using argmax instead of center-of-mass to avoid center bias
        # pulling the CoM toward image center for peripheral fixations.
        pred_idx = np.unravel_index(pred[b, 0].argmax(), pred[b, 0].shape)
        pred_point = pred_idx

        # GT gaze: argmax (returns row, col = y, x)
        gt_idx = np.unravel_index(target[b, 0].argmax(), target[b, 0].shape)

        # Construct 3D rays: [y - center, x - center, focal_length]
        r1 = np.array([pred_point[0] - center, pred_point[1] - center, d])
        r2 = np.array([gt_idx[0] - center,     gt_idx[1] - center,     d])

        # Angle between rays (numerically stable via atan2)
        angle = math.atan2(
            np.linalg.norm(np.cross(r1, r2)),
            np.dot(r1, r2),
        )
        aae_list.append(math.degrees(angle))

    return np.mean(aae_list)


def compute_pixel_distance(pred_heatmap, target_heatmap, pred_method='com'):
    """
    Compute Euclidean pixel distance between predicted and GT gaze points.
    
    Default: center-of-mass for pred, argmax for GT (consistent with AUC/AAE).
    Use pred_method='argmax' to align with distance-based F1.
    
    Args:
        pred_heatmap: Predicted heatmap [batch, 1, H, W]
        target_heatmap: Ground truth heatmap [batch, 1, H, W]
        pred_method: 'com' (default, matches AUC/AAE) or 'argmax' (matches F1)
        
    Returns:
        Mean pixel distance across batch
    """
    pred_points, gt_points = extract_gaze_points(
        pred_heatmap, target_heatmap, 
        pred_method=pred_method, gt_method='argmax'
    )
    distances = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis=1))
    return np.mean(distances)


# =============================================================================
# Efficiency Metrics (one-time computation)
# =============================================================================

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def compute_flops(model, device='cpu'):
    """
    Compute FLOPs using thop library.
    
    Returns GFLOPs (giga floating-point operations).
    Install: pip install thop
    """
    try:
        from thop import profile
    except ImportError:
        print("thop not installed. Run: pip install thop")
        return None
    
    model = model.to(device)
    model.eval()
    
    frame_t = torch.randn(1, 3, 300, 300).to(device)
    frame_t_minus_1 = torch.randn(1, 3, 300, 300).to(device)
    gaze_t_minus_1 = torch.tensor([[150.0, 150.0]]).to(device)
    
    with torch.no_grad():
        flops, params = profile(model, inputs=(frame_t, frame_t_minus_1, gaze_t_minus_1), verbose=False)
    
    return flops / 1e9  # GFLOPs


def compute_inference_time(model, input_shape=(1, 3, 300, 300), device='cpu', num_runs=100):
    """
    Measure inference time per frame.
    
    For GPU timing, uses cuda.Event for accurate measurement.
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    frame_t = torch.randn(input_shape).to(device)
    frame_t_minus_1 = torch.randn(input_shape).to(device)
    gaze_t_minus_1 = torch.tensor([[150.0, 150.0]]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(frame_t, frame_t_minus_1, gaze_t_minus_1)
    
    if device != 'cpu' and torch.cuda.is_available():
        # GPU timing with cuda events (accurate)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                _ = model(frame_t, frame_t_minus_1, gaze_t_minus_1)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
    else:
        # CPU timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(frame_t, frame_t_minus_1, gaze_t_minus_1)
                end = time.perf_counter()
                times.append((end - start) * 1000)
    
    return np.mean(times)


# =============================================================================
# GPU Heatmap Generation (Training Speedup)
# =============================================================================

def make_gaussian_kernel(sigma, kernel_size=None):
    """
    Pre-compute a 2D Gaussian kernel for GPU heatmap generation.
    
    Call once before the training loop, move to GPU with .to(device).
    
    Args:
        sigma: Gaussian sigma in pixels
        kernel_size: Kernel size (auto: 6*sigma + 1, must be odd)
        
    Returns:
        kernel: Tensor [1, 1, k, k] for use with F.conv2d
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]


def create_heatmap_gpu(gaze_coords, kernel, size=300):
    """
    Create a batch of heatmaps on GPU using pre-computed Gaussian kernel.
    
    Drop-in replacement for the CPU create_gaze_heatmap() loop in train.py.
    Expected speedup: 3-5x on the heatmap generation bottleneck.
    
    Args:
        gaze_coords: [B, 2] gaze coordinates in target resolution (300×300)
        kernel: Pre-computed Gaussian kernel from make_gaussian_kernel()
        size: Heatmap spatial size
        
    Returns:
        heatmaps: [B, 1, size, size] normalized to [0, 1], on same device as input
    """
    import torch.nn.functional as F
    
    B = gaze_coords.shape[0]
    device = gaze_coords.device
    
    hm = torch.zeros(B, 1, size, size, device=device)
    gx = gaze_coords[:, 0].long().clamp(0, size - 1)
    gy = gaze_coords[:, 1].long().clamp(0, size - 1)
    
    for i in range(B):
        hm[i, 0, gy[i], gx[i]] = 1.0
    
    pad = kernel.shape[-1] // 2
    kernel_dev = kernel.to(device)
    hm = F.conv2d(hm, kernel_dev, padding=pad)
    
    # Normalize each heatmap to [0, 1]
    hm_max = hm.amax(dim=(2, 3), keepdim=True)
    hm = hm / (hm_max + 1e-7)
    
    return hm


# =============================================================================
# Combined Metrics Class
# =============================================================================

class GazeLiteMetrics:
    """
    Convenience class to compute all metrics at once.
    All gaze point extraction is done internally from heatmaps.
    """

    def __init__(self, image_size=300, gt_sigma=16.4, fov_degrees=None):
        self.image_size  = image_size
        self.gt_sigma    = gt_sigma
        self.fov_degrees = fov_degrees  # None → EGTEA 60° default; 78 for Ego-Exo-4D

    def compute_all(self, pred_heatmap, target_heatmap):
        """
        Compute all gaze prediction metrics from heatmaps.

        Args:
            pred_heatmap: Predicted heatmap [batch, 1, H, W]
            target_heatmap: Ground truth heatmap [batch, 1, H, W]

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Core gaze metrics (center-of-mass pred, argmax GT)
        metrics['auc'] = compute_auc(pred_heatmap, target_heatmap)
        metrics['nss'] = compute_nss(pred_heatmap, target_heatmap)
        metrics['aae'] = compute_aae(pred_heatmap, target_heatmap,
                                     self.image_size, fov_degrees=self.fov_degrees)
        metrics['pixel_distance'] = compute_pixel_distance(pred_heatmap, target_heatmap)

        # Distance-based F1 (argmax pred, argmax GT)
        dist_f1 = compute_distance_f1(pred_heatmap, target_heatmap, sigma=self.gt_sigma)
        metrics.update(dist_f1)

        # Pixel-level F1 (Lai et al. compatible)
        pixel_f1 = compute_pixel_f1(pred_heatmap, target_heatmap)
        metrics.update(pixel_f1)

        return metrics