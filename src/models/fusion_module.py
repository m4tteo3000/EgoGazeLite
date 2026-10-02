"""
Fusion module for combining saliency and attention transition heatmaps.

Two modes, switchable via config:
    - "huang":    Conv stack (Huang et al. 2018) — learned nonlinear spatial fusion
    - "residual": Residual refinement — G_s as base, learn correction from G_a

"""

import torch
import torch.nn as nn


class HuangFusion(nn.Module):
    """
    Late fusion from Huang et al. (ECCV 2018), Section 3.7.
    
    Concatenates G_s and G_a, runs through a small conv stack.
    Returns raw logits — sigmoid applied by GatedFusion wrapper.
    
    Architecture: 4 conv layers (2→32→32→8→1).
    ~10K parameters.
    """
    
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            # No sigmoid — returns logits
        )
    
    def forward(self, g_s, g_a):
        g_s_norm = torch.sigmoid(g_s)              # normalise logits → [0,1] to match G_a scale
        x = torch.cat([g_s_norm, g_a], dim=1)      # [B, 2, 300, 300] — both in probability space
        logits = self.fusion(x)                    # [B, 1, 300, 300]
        return logits


class ResidualFusion(nn.Module):
    """
    Residual refinement fusion.

    Uses G_s (saliency) as the base prediction and learns a residual correction
    informed by G_a (attention transition).

    Expects G_s already in logit space (saliency_decoder.return_logits=True,
    enforced by GazeLite.__init__ when fusion_mode='residual').

    Returns logits: G_s + residual(G_s, G_a)
    Sigmoid applied by GatedFusion wrapper.

    ~10K parameters (same ballpark as Huang).
    """

    def __init__(self):
        super().__init__()
        self.residual_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            # No activation — raw residual, can be positive or negative
        )

    def forward(self, g_s, g_a):
        g_s_norm = torch.sigmoid(g_s)              # normalise logits → [0,1] to match G_a scale
        x = torch.cat([g_s_norm, g_a], dim=1)      # [B, 2, 300, 300] — both in probability space
        residual = self.residual_net(x)            # [B, 1, 300, 300]

        # Residual added back in logit space — output stays as logits
        logits = g_s + residual                    # [B, 1, 300, 300]
        return logits


class GatedFusion(nn.Module):
    """
    Config-switchable fusion module.

    Wraps HuangFusion and ResidualFusion behind a unified interface.
    Both modes return logits; sigmoid is applied here for metrics/visualization.
    The loss function receives logits directly via return_logits=True.

    Config usage:
        stage3:
            fusion_mode: "huang"    # or "residual"
    """

    MODES = {
        'huang': HuangFusion,
        'residual': ResidualFusion,
    }

    def __init__(self, mode='huang'):
        super().__init__()
        
        if mode not in self.MODES:
            raise ValueError(
                f"Unknown fusion mode '{mode}'. "
                f"Options: {list(self.MODES.keys())}"
            )
        
        self.mode = mode
        self.fusion = self.MODES[mode]()
    
    def forward(self, g_s, g_a, return_logits=False):
        """
        Args:
            g_s: Saliency heatmap  [batch, 1, 300, 300]
            g_a: Attention heatmap [batch, 1, 300, 300]
            return_logits: If True, return raw logits (for loss computation).
                           If False, return probabilities (for metrics/viz).
            
        Returns:
            g_t:         Fused gaze heatmap [batch, 1, 300, 300]
            gaze_coords: Gaze position (x, y) [batch, 2]
        """
        output = self.fusion(g_s, g_a)  # logits

        if return_logits:
            # Return logits for loss — sigmoid will be in the loss function
            g_t_for_loss = output
        else:
            # Apply sigmoid to get probabilities for metrics/viz
            g_t_for_loss = torch.sigmoid(output)
        
        # Argmax on probabilities for gaze coordinates
        g_t_prob = torch.sigmoid(output)
        batch_size = g_t_prob.shape[0]
        h, w = g_t_prob.shape[2], g_t_prob.shape[3]
        g_t_flat = g_t_prob.view(batch_size, -1)
        max_idx = g_t_flat.argmax(dim=1)
        
        gaze_y = max_idx // w
        gaze_x = max_idx % w
        gaze_coords = torch.stack([gaze_x, gaze_y], dim=1).float()
        
        return g_t_for_loss, gaze_coords