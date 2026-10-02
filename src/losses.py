"""
Loss functions for staged GazeLite training.
Stage 1: Saliency decoder
Stage 2: Attention transition (fixation + LSTM)
Stage 3: Gated fusion

v2: FusionLoss uses binary_cross_entropy_with_logits for AMP compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyLoss(nn.Module):
    """
    Modified BCE loss for saliency/gaze heatmap prediction.
    Downweights pixels near gaze point to handle noisy gaze measurements.
    From Huang et al. Eq. 6.
    """
    
    def __init__(self, image_size=300):
        super().__init__()
        self.image_size = image_size
        
    def forward(self, pred, target, gaze_coords):
        batch_size = pred.shape[0]
        
        y_coords = torch.arange(self.image_size, device=pred.device).float()
        x_coords = torch.arange(self.image_size, device=pred.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        weights = []
        for i in range(batch_size):
            gx, gy = gaze_coords[i]
            dist = torch.sqrt((xx - gx)**2 + (yy - gy)**2)
            dist_normalized = dist / self.image_size
            weight = 1 + dist_normalized
            weights.append(weight)
        
        weights = torch.stack(weights).unsqueeze(1)
        
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = (weights * bce).mean()
        
        return weighted_bce

    
class SaliencyKLDLoss(nn.Module):
    """
    KL-Divergence loss with temperature-scaled softmax for saliency prediction.
    Following Lai et al. "In the Eye of the Transformer" (IJCV 2024).
    Expects raw logits from the decoder (return_logits=True in SaliencyDecoder).
    """
    
    def __init__(self, temperature=2.0, eps=1e-7):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, pred, target):
        batch_size = pred.shape[0]

        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        
        pred_dist = torch.softmax(pred_flat / self.temperature, dim=1)
        target_dist = target_flat / (target_flat.sum(dim=1, keepdim=True) + self.eps)
        
        kld = (target_dist * torch.log((target_dist + self.eps) / (pred_dist + self.eps))).sum(dim=1)
        
        return kld.mean()

    
class SaliencyLogitsLoss(nn.Module):
    """
    Modified BCE loss operating on raw logits (numerically stable).
    Uses BCEWithLogitsLoss which combines sigmoid + BCE in one step.
    Same distance-weighted approach as SaliencyLoss (Huang et al. Eq. 6).
    """
    
    def __init__(self, image_size=300):
        super().__init__()
        self.image_size = image_size
        
    def forward(self, pred, target, gaze_coords):
        batch_size = pred.shape[0]
        
        y_coords = torch.arange(self.image_size, device=pred.device).float()
        x_coords = torch.arange(self.image_size, device=pred.device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        weights = []
        for i in range(batch_size):
            gx, gy = gaze_coords[i]
            dist = torch.sqrt((xx - gx)**2 + (yy - gy)**2)
            dist_normalized = dist / self.image_size
            weight = 1 + dist_normalized
            weights.append(weight)
        
        weights = torch.stack(weights).unsqueeze(1)
        
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (weights * bce).mean()
        
        return weighted_bce


class AttentionTransitionLoss(nn.Module):
    """
    MSE loss for LSTM channel weight prediction.
    From Huang et al. Eq. 7.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_weights, target_weights):
        return self.mse(pred_weights, target_weights)


class FusionLoss(nn.Module):
    """
    Distance-weighted BCE loss for final gaze heatmap prediction.
    
    v2: Operates on logits (uses binary_cross_entropy_with_logits) for
    AMP compatibility. The fusion module (huang/residual modes) now returns
    logits; sigmoid is handled inside the loss function.
    
    For gated mode (which returns probabilities), automatically converts
    to logits before computing the loss.
    """
    
    def __init__(self, use_distance_weights=True, image_size=300):
        super().__init__()
        self.use_distance_weights = use_distance_weights
        self.image_size = image_size
        
    def forward(self, pred, target, gaze_coords=None):
        """
        Args:
            pred: Fused heatmap logits [batch, 1, 300, 300] (from huang/residual)
                  or probabilities (from gated mode)
            target: Ground truth heatmap [batch, 1, 300, 300] (probabilities)
            gaze_coords: Ground truth gaze (x, y) [batch, 2]
        """
        if self.use_distance_weights and gaze_coords is not None:
            batch_size = pred.shape[0]
            
            y_coords = torch.arange(self.image_size, device=pred.device).float()
            x_coords = torch.arange(self.image_size, device=pred.device).float()
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            weights = []
            for i in range(batch_size):
                gx, gy = gaze_coords[i]
                dist = torch.sqrt((xx - gx)**2 + (yy - gy)**2)
                dist_normalized = dist / self.image_size
                weight = 1 + dist_normalized
                weights.append(weight)
            
            weights = torch.stack(weights).unsqueeze(1)
            
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            return (weights * bce).mean()
        else:
            return F.binary_cross_entropy_with_logits(pred, target)


