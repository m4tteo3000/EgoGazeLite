"""
EfficientNet-Lite4 backbone for feature extraction.
Supports multi-scale feature extraction for skip connections.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-Lite4 backbone for feature extraction.
    Returns multi-scale feature maps for skip connections in decoder.
    
    Feature pyramid for 300x300 input:
        - f1: 75x75x32   (index 1, stride 4)
        - f2: 38x38x56   (index 2, stride 8)
        - f3: 19x19x160  (index 3, stride 16)
        - f4: 10x10x448  (index 4, stride 32)  <- main output
    """
    
    def __init__(self, pretrained=True, multi_scale=True):
        """
        Args:
            pretrained: Whether to load ImageNet pretrained weights
            multi_scale: If True, return all scales for skip connections.
                        If False, return only final feature map (backward compatible).
        """
        super().__init__()
        
        self.multi_scale = multi_scale
        
        if multi_scale:
            # Return multiple feature scales for skip connections
            self.backbone = timm.create_model(
                'tf_efficientnet_lite4.in1k',
                pretrained=pretrained,
                features_only=True,
                out_indices=[1, 2, 3, 4]  # Skip index 0 (too large, 150x150)
            )
            # Feature info for each output
            # Index 1: 75x75x32
            # Index 2: 38x38x56
            # Index 3: 19x19x160
            # Index 4: 10x10x448
            self.feature_channels = {
                'f1': 32,   # 75x75
                'f2': 56,   # 38x38
                'f3': 160,  # 19x19
                'f4': 448,  # 10x10
            }
        else:
            # Backward compatible: only return final features
            self.backbone = timm.create_model(
                'tf_efficientnet_lite4.in1k',
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1]
            )
        
        # Get output channels for the main (final) feature map
        self.out_channels = self.backbone.feature_info[-1]['num_chs']

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, 3, 300, 300]
            
        Returns:
            If multi_scale=True:
                Dict with keys 'f1', 'f2', 'f3', 'f4' containing feature maps
            If multi_scale=False:
                Feature map [batch, 448, 10, 10]
        """
        features = self.backbone(x)
        
        if self.multi_scale:
            return {
                'f1': features[0],  # [batch, 32, 75, 75]
                'f2': features[1],  # [batch, 56, 38, 38]
                'f3': features[2],  # [batch, 160, 19, 19]
                'f4': features[3],  # [batch, 448, 10, 10]
            }
        else:
            return features[0]
    
    def get_skip_channels(self):
        """Return channel counts for skip connections."""
        if self.multi_scale:
            return self.feature_channels
        else:
            raise ValueError("Skip channels only available when multi_scale=True")