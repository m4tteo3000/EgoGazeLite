"""
Saliency decoder module with skip connections and center bias.
Upsamples fused features to produce saliency heatmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyDecoder(nn.Module):
    """
    Decodes fused features (F_t + ΔF) into a saliency heatmap.
    
    Improvements over baseline:
    1. Skip connections from intermediate backbone features for spatial detail
    2. Learnable center bias prior for egocentric gaze patterns
    
    Architecture:
        Input: f4 (10x10x448) + delta_f (10x10x448) = 10x10x896
        + skip from f3 (19x19x160) -> used at 20x20
        + skip from f2 (38x38x56)  -> used at 40x40  
        + skip from f1 (75x75x32)  -> used at 80x80
        Output: 300x300x1 saliency map
    """
    
    def __init__(self, in_channels=896, skip_channels=None, use_center_bias=True, return_logits=False, center_bias_sigma=0.4):
        """
        Args:
            in_channels: Input channels (448 + 448 = 896 for concatenated F_t and ΔF)
            skip_channels: Dict with channels for each skip connection level
                          e.g., {'f1': 32, 'f2': 56, 'f3': 160}
                          If None, skip connections are disabled (backward compatible)
            use_center_bias: Whether to add learnable center bias prior
        """
        super().__init__()
        
        self.use_skip = skip_channels is not None
        self.use_center_bias = use_center_bias
        
        # Default skip channels for EfficientNet-Lite4
        if skip_channels is None:
            skip_channels = {'f1': 0, 'f2': 0, 'f3': 0}
        
        # Fusion: 896 → 256
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Upsample block 1: 10→20, with skip from f3 (19x19x160, resized to 20x20)
        # Input: 256 + 160 (skip) = 416 → 128
        skip3_ch = skip_channels.get('f3', 0)
        self.up1 = self._upsample_block(256 + skip3_ch, 128)
        
        # Upsample block 2: 20→40, with skip from f2 (38x38x56, resized to 40x40)
        # Input: 128 + 56 (skip) = 184 → 64
        skip2_ch = skip_channels.get('f2', 0)
        self.up2 = self._upsample_block(128 + skip2_ch, 64)
        
        # Upsample block 3: 40→80, with skip from f1 (75x75x32, resized to 80x80)
        # Input: 64 + 32 (skip) = 96 → 32
        skip1_ch = skip_channels.get('f1', 0)
        self.up3 = self._upsample_block(64 + skip1_ch, 32)
        
        # Upsample block 4: 80→160 (no skip connection at this level)
        self.up4 = self._upsample_block(32, 16)
        
        # Final: 160→320, then Conv1×1 to 1 channel
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        # Center bias prior
        if use_center_bias:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 300, 300))
            self._init_center_bias(sigma=center_bias_sigma)
        
        # Output mode: logits (for KLD loss) or probabilities (for BCE loss)
        self.return_logits = return_logits  
    
    def _upsample_block(self, in_ch, out_ch):
        """Helper to create conv block (upsampling done separately via interpolate)."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_center_bias(self, sigma=0.4):
        """
        Initialize center bias with 2D Gaussian.
        Egocentric gaze tends toward image center.
        
        Args:
            sigma: Standard deviation of Gaussian (in normalized coords).
                   0.4 gives moderate center bias, covering roughly middle 60% of image.
        """
        y = torch.linspace(-1, 1, 300)
        x = torch.linspace(-1, 1, 300)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 2D Gaussian centered at (0, 0)
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Scale to reasonable initial magnitude (will be learned)
        # Using 0.5 so sigmoid(0 + 0.5) ≈ 0.62 at center, sigmoid(0 + 0) ≈ 0.5 at edges
        gaussian = gaussian * 0.5
        
        self.center_bias.data = gaussian.unsqueeze(0).unsqueeze(0)
    
    def forward(self, f_t, delta_f, skip_features=None):
        """
        Args:
            f_t: Current frame features [batch, 448, 10, 10]
            delta_f: Temporal difference [batch, 448, 10, 10]
            skip_features: Optional dict with skip connection features
                          {'f1': [B,32,75,75], 'f2': [B,56,38,38], 'f3': [B,160,19,19]}
            
        Returns:
            Saliency map [batch, 1, 300, 300]
        """
        # Concatenate F_t and ΔF
        x = torch.cat([f_t, delta_f], dim=1)  # [batch, 896, 10, 10]
        
        # Fusion
        x = self.fusion(x)  # [batch, 256, 10, 10]
        
        # === Upsample block 1: 10 → 20 ===
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # [batch, 256, 20, 20]
        
        if self.use_skip and skip_features is not None and 'f3' in skip_features:
            # Resize f3 from 19x19 to 20x20 and concatenate
            skip3 = F.interpolate(skip_features['f3'], size=(20, 20), 
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip3], dim=1)  # [batch, 256+160, 20, 20]
        
        x = self.up1(x)  # [batch, 128, 20, 20]
        
        # === Upsample block 2: 20 → 40 ===
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # [batch, 128, 40, 40]
        
        if self.use_skip and skip_features is not None and 'f2' in skip_features:
            # Resize f2 from 38x38 to 40x40 and concatenate
            skip2 = F.interpolate(skip_features['f2'], size=(40, 40),
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip2], dim=1)  # [batch, 128+56, 40, 40]
        
        x = self.up2(x)  # [batch, 64, 40, 40]
        
        # === Upsample block 3: 40 → 80 ===
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # [batch, 64, 80, 80]
        
        if self.use_skip and skip_features is not None and 'f1' in skip_features:
            # Resize f1 from 75x75 to 80x80 and concatenate
            skip1 = F.interpolate(skip_features['f1'], size=(80, 80),
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip1], dim=1)  # [batch, 64+32, 80, 80]
        
        x = self.up3(x)  # [batch, 32, 80, 80]
        
        # === Upsample block 4: 80 → 160 ===
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x)  # [batch, 16, 160, 160]
        
        # === Final: 160 → 320 → 300 ===
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.final_conv(x)  # [batch, 1, 320, 320]
        
        # Resize to exact output size
        x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
        
        # Add center bias before activation (if enabled)
        if self.use_center_bias:
            x = x + self.center_bias

        # Final activation
        # return_logits=True skips sigmoid, for use with KLD + softmax loss
        if self.return_logits:
            return x  # [batch, 1, 300, 300] raw logits
        else:
            return torch.sigmoid(x)  # [batch, 1, 300, 300] probabilities
