"""
Attention weight application module.
Applies channel weights to feature map to produce attention heatmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWeightApplication(nn.Module):
    """
    Applies channel weights to feature map to produce attention heatmap.
    G_a^t = Sigmoid(Upsample(sum_c(w_t[c] * F_t[c])))
    """
    
    def __init__(self, output_size=300):
        """
        Args:
            output_size: Output spatial size (300)
        """
        super().__init__()
        
        self.output_size = output_size

    def forward(self, f_t, w_t):
        """
        Args:
            f_t: Current frame features [batch, 448, 10, 10]
            w_t: Channel weights [batch, 448]
            
        Returns:
            Attention heatmap G_a^t [batch, 1, 300, 300]
        """
        # Reshape w_t for broadcasting: [batch, 448] → [batch, 448, 1, 1]
        w_t = w_t.unsqueeze(-1).unsqueeze(-1)  # [batch, 448, 1, 1]
        
        # Channel-wise weighted sum: sum_c(w_t[c] * F_t[c])
        weighted = w_t * f_t              # [batch, 448, 10, 10]
        attention = weighted.sum(dim=1, keepdim=True)  # [batch, 1, 10, 10]
        
        # Upsample to output size
        attention = F.interpolate(
            attention, 
            size=(self.output_size, self.output_size), 
            mode='bilinear', 
            align_corners=False
        )  # [batch, 1, 300, 300]
        
        # Sigmoid to normalize to [0, 1]
        attention = torch.sigmoid(attention)
        
        return attention
    
