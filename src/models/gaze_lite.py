"""
GazeLite: Complete gaze prediction model.
Assembles all modules into a single end-to-end model.

"""

import torch
import torch.nn as nn

from .backbone import EfficientNetBackbone
from .temporal_diff import TemporalDifference
from .saliency_decoder import SaliencyDecoder
from .fusion_module import GatedFusion
from .attention_transition.channel_weights import ChannelWeightExtractor
from .attention_transition.lstm_gated import LSTMGatedUpdate
from .attention_transition.attention_weight_application import AttentionWeightApplication

# Fixation detection constants
FIXATION_DISPERSION_THRESHOLD = 16.0   # raised from 8px — Aria wearable noise floor ~1-2 deg
FIXATION_IDT_WINDOW = 3                 # 300ms at 10fps (time-equivalent to EGTEA's 7 frames@24fps)


class GazeLite(nn.Module):
    """
    Complete gaze prediction model.
    Combines saliency prediction and attention transition pathways.
    """
    
    def __init__(self, pretrained_backbone=True, use_skip_connections=True,
                 use_center_bias=True, center_bias_sigma=0.4,
                 fusion_mode='residual'):
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        
        # Backbone
        self.backbone = EfficientNetBackbone(
            pretrained=pretrained_backbone,
            multi_scale=use_skip_connections
        )
        
        # Saliency path
        self.temporal_diff = TemporalDifference(num_channels=448)
        
        if use_skip_connections:
            self.saliency_decoder = SaliencyDecoder(
                in_channels=896,
                skip_channels=self.backbone.get_skip_channels(),
                use_center_bias=use_center_bias,
                center_bias_sigma=center_bias_sigma
            )
        else:
            self.saliency_decoder = SaliencyDecoder(
                in_channels=896,
                skip_channels=None,
                use_center_bias=use_center_bias,
                center_bias_sigma=center_bias_sigma
            )
        
        # Attention transition path
        self.channel_weight_extractor = ChannelWeightExtractor(
            input_size=300, feature_size=10, roi_size=3
        )
        self.lstm_gated = LSTMGatedUpdate(
            weight_dim=448, hidden_dim=448, num_layers=2
        )
        self.attention_weight_app = AttentionWeightApplication(output_size=300)
        
        # Fusion
        self.gated_fusion = GatedFusion(mode=fusion_mode)

        # Fusion modules (huang, residual) consume G_s as logits.
        # Enforce this here so the model is self-consistent regardless of
        # how train.py configures the loss function.
        # Both fusion modes (huang, residual) consume G_s as logits.
        self.saliency_decoder.return_logits = True

    def compute_fixation_state(self, gaze_history, method='idt'):
        """
        Compute fixation state from gaze coordinate history using I-DT
        (Dispersion Threshold Identification).

        Args:
            gaze_history: List of recent gaze tensors [batch, 2], newest last.
            method: Only 'idt' is supported.

        Returns:
            f_t: [batch, 1] -- 1.0 = fixation, 0.0 = saccade
        """
        if len(gaze_history) < 2:
            batch_size = gaze_history[0].shape[0]
            return torch.ones(batch_size, 1, device=gaze_history[0].device)

        if method == 'idt':
            window_size = min(len(gaze_history), FIXATION_IDT_WINDOW)
            window = gaze_history[-window_size:]
            
            if window_size < 2:
                batch_size = gaze_history[0].shape[0]
                return torch.ones(batch_size, 1, device=gaze_history[0].device)
            
            points = torch.stack(window, dim=0)
            
            max_xy = points.max(dim=0).values
            min_xy = points.min(dim=0).values
            dispersion = (max_xy - min_xy).sum(dim=1, keepdim=True)
            
            f_t = (dispersion < FIXATION_DISPERSION_THRESHOLD).float()
        
        else:
            raise ValueError(f"Unknown fixation method: {method}")

        return f_t

    def _extract_features(self, frame):
        """Extract features from a frame."""
        if self.use_skip_connections:
            features = self.backbone(frame)
            f4 = features['f4']
            skip_features = {
                'f1': features['f1'],
                'f2': features['f2'],
                'f3': features['f3'],
            }
            return f4, skip_features
        else:
            f4 = self.backbone(frame)
            return f4, None

    def forward(self, frame_t, frame_t_minus_1, gaze_t_minus_1,
                hidden=None, gaze_history=None, fixation_method='idt',
                return_intermediates=False, return_logits=False):
        """
        Full forward pass.
        
        Args:
            frame_t: Current frame [batch, 3, 300, 300]
            frame_t_minus_1: Previous frame [batch, 3, 300, 300]
            gaze_t_minus_1: Previous gaze coordinates [batch, 2]
            hidden: LSTM hidden state (h, c) or None
            gaze_history: List of recent gaze tensors for fixation detection
            fixation_method: Fixation detection method (currently 'idt')
            return_intermediates: If True, returns dict with g_s, g_a, g_t
            return_logits: If True, g_t is logits (for loss). If False, probabilities.
            
        Returns:
            g_t: Gaze heatmap [batch, 1, 300, 300] (logits or probabilities)
            gaze_coords: Predicted gaze (x, y) [batch, 2]
            hidden: Updated LSTM hidden state
            fixation: Computed fixation state [batch, 1]
            intermediates: (only if return_intermediates) dict with g_s, g_a, g_t
        """
        f_t, skip_features_t = self._extract_features(frame_t)
        f_t_minus_1, _ = self._extract_features(frame_t_minus_1)
        
        # Saliency Path
        delta_f = self.temporal_diff(f_t, f_t_minus_1)
        g_s = self.saliency_decoder(f_t, delta_f, skip_features=skip_features_t)
        
        # Attention Transition Path
        if gaze_history is not None and len(gaze_history) >= 2:
            fixation = self.compute_fixation_state(gaze_history, method=fixation_method)
        else:
            batch_size = frame_t.shape[0]
            fixation = torch.ones(batch_size, 1, device=frame_t.device)
        
        w_t_minus_1 = self.channel_weight_extractor(f_t_minus_1, gaze_t_minus_1)
        w_t, hidden = self.lstm_gated(w_t_minus_1, fixation, hidden)
        g_a = self.attention_weight_app(f_t, w_t)
        
        # Fusion — pass return_logits so loss gets logits, metrics get probabilities
        g_t, gaze_coords = self.gated_fusion(g_s, g_a, return_logits=return_logits)
        
        if return_intermediates:
            # Intermediates always as probabilities for visualization
            g_t_prob = g_t if not return_logits else torch.sigmoid(g_t)
            intermediates = {'g_s': g_s, 'g_a': g_a, 'g_t': g_t_prob}
            return g_t, gaze_coords, hidden, fixation, intermediates
        
        return g_t, gaze_coords, hidden, fixation
    
    @torch.no_grad()
    def precompute_sequence_features(self, frames: torch.Tensor) -> dict:
        """
        Pre-compute frozen Stage 1 outputs for an entire sequence in one
        batched backbone pass.  Replaces T sequential _extract_features calls
        with a single [B*T, C, H, W] forward — the backbone is ~90% of Stage
        2/3 per-step cost.

        Call this once at the top of each training / validation step, then
        index into the returned tensors instead of calling model.forward().

        Args:
            frames: [B, T, C, H, W]

        Returns dict:
            f4:  [B, T, C_f, h, w]   backbone features per frame
            g_s: [B, T, 1,   H, W]   saliency maps per frame
        """
        B, T, C, H, W = frames.shape

        # ── 1. Backbone on all B*T frames at once ────────────────────────
        flat = frames.reshape(B * T, C, H, W)

        if self.use_skip_connections:
            raw = self.backbone(flat)
            f4_flat   = raw['f4']
            skip_flat = {'f1': raw['f1'], 'f2': raw['f2'], 'f3': raw['f3']}
        else:
            f4_flat   = self.backbone(flat)
            skip_flat = None

        _, Cf, Hf, Wf = f4_flat.shape

        # ── 2. Temporal diff for all adjacent pairs ───────────────────────
        f4 = f4_flat.reshape(B, T, Cf, Hf, Wf)

        f4_curr = f4[:, 1:].reshape(B * (T - 1), Cf, Hf, Wf)
        f4_prev = f4[:, :-1].reshape(B * (T - 1), Cf, Hf, Wf)
        delta_pairs = self.temporal_diff(f4_curr, f4_prev)   # [B*(T-1), Cf, Hf, Wf]

        # t=0 has no previous frame — pad with zeros
        delta_0 = torch.zeros(B, 1, Cf, Hf, Wf,
                               device=frames.device, dtype=f4_flat.dtype)
        delta_f = torch.cat(
            [delta_0, delta_pairs.reshape(B, T - 1, Cf, Hf, Wf)], dim=1
        ).reshape(B * T, Cf, Hf, Wf)                         # [B*T, Cf, Hf, Wf]

        # ── 3. Saliency decoder on all B*T frames ─────────────────────────
        g_s_flat = self.saliency_decoder(f4_flat, delta_f, skip_features=skip_flat)
        _, Cs, Hs, Ws = g_s_flat.shape

        return {
            'f4':  f4,                                         # [B, T, Cf, Hf, Wf]
            'g_s': g_s_flat.reshape(B, T, Cs, Hs, Ws),       # [B, T, 1, H, W]
        }

    def forward_saliency_only(self, frame_t, frame_t_minus_1):
        """Forward pass through saliency path only (Stage 1 training)."""
        f_t, skip_features_t = self._extract_features(frame_t)
        f_t_minus_1, _ = self._extract_features(frame_t_minus_1)
        
        delta_f = self.temporal_diff(f_t, f_t_minus_1)
        g_s = self.saliency_decoder(f_t, delta_f, skip_features=skip_features_t)
        
        return g_s