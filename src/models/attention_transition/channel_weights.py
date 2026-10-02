"""
Channel weight extractor using RoI-Align.
Extracts attention weights from feature map around gaze location.
"""

import torch
import torch.nn as nn
from torchvision.ops import roi_align

class ChannelWeightExtractor(nn.Module):
    """
    Extracts channel weights from feature map around gaze location.
    Uses RoI-Align to crop region around gaze point.
    """
    
    def __init__(self, input_size=300, feature_size=10, roi_size=3):
        """
        Args:
            input_size: Original image size (300)
            feature_size: Feature map spatial size (10)
            roi_size: Size of RoI box to extract (3x3)
        """
        super().__init__()
        
        self.scale = feature_size / input_size  # 10/300 = 0.0333
        self.roi_size = roi_size
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, f_t_minus_1, gaze_t_minus_1):
        """
        Args:
            f_t_minus_1: Previous frame features [batch, 448, 10, 10]
            gaze_t_minus_1: Previous gaze coordinates [batch, 2] in 300x300 space
            
        Returns:
            Channel weights w_{t-1} [batch, 448]
        """
        batch_size = f_t_minus_1.shape[0]
        
        # Scale gaze from 300x300 to 10x10 space
        gaze_scaled = gaze_t_minus_1 * self.scale  # [batch, 2]
        
        # Create RoI boxes: [batch_idx, x1, y1, x2, y2]
        half = self.roi_size / 2  # 1.5
        boxes = []
        for i in range(batch_size):
            x, y = gaze_scaled[i]
            box = torch.tensor([i, x - half, y - half, x + half, y + half], 
                               device=f_t_minus_1.device)
            boxes.append(box)
        boxes = torch.stack(boxes)  # [batch, 5]
        
        # RoI-Align: extract 3x3 patch for each sample
        roi_features = roi_align(
            f_t_minus_1, 
            boxes, 
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,  # Already in feature space
            aligned=True
        )  # [batch, 448, 3, 3]
        
        # Global average pool to get channel weights
        w = self.pool(roi_features)  # [batch, 448, 1, 1]
        w = w.flatten(1)              # [batch, 448]
        
        return w



