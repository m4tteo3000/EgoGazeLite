"""
Temporal difference module.
Computes normalized difference between consecutive feature maps.
"""

import torch
import torch.nn as nn

class TemporalDifference(nn.Module):
    """
    Computes normalized difference between consecutive feature maps.
    ΔF = BatchNorm(F_t - F_{t-1})
    """
    
    def __init__(self, num_channels=448):
        """
        Args:
            num_channels: Number of feature channels (448 for EfficientNet-Lite4)
        """
        super().__init__()
        
        self.bn = nn.BatchNorm2d(num_channels)
    
    def forward(self, f_t, f_t_minus_1):
        """
        Args:
            f_t: Current frame features [batch, 448, 10, 10]
            f_t_minus_1: Previous frame features [batch, 448, 10, 10]
            
        Returns:
            Normalized difference [batch, 448, 10, 10]
        """
        diff = f_t - f_t_minus_1
        return self.bn(diff)