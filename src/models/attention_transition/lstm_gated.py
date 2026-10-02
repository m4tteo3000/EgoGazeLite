"""
LSTM-based weight predictor with gated update.
Predicts attention transition and combines with fixation state.
"""

import torch
import torch.nn as nn

class LSTMGatedUpdate(nn.Module):
    """
    LSTM-based channel weight predictor with gated update.
    Predicts next attention weights and blends with current based on fixation state.
    """
    
    def __init__(self, weight_dim=448, hidden_dim=448, num_layers=3):
        """
        Args:
            weight_dim: Dimension of channel weights (448)
            hidden_dim: LSTM hidden size (256)
            num_layers: Number of LSTM layers (3)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project weights to LSTM input size
        self.fc_in = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3-layer LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Project back to weight dimension
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, weight_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, w_t_minus_1, f_t, hidden=None):
        """
        Args:
            w_t_minus_1: Previous channel weights [batch, 448]
            f_t: Fixation probability [batch, 1]
            hidden: LSTM hidden state (h, c) or None for initial
            
        Returns:
            w_t: Updated channel weights [batch, 448]
            hidden: Updated LSTM hidden state
        """
        batch_size = w_t_minus_1.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                           device=w_t_minus_1.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
                           device=w_t_minus_1.device)
            hidden = (h, c)
        
        # Project to LSTM space
        x = self.fc_in(w_t_minus_1)  # [batch, 256]
        x = x.unsqueeze(1)            # [batch, 1, 256] - single timestep
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # [batch, 1, 256]
        lstm_out = lstm_out.squeeze(1)            # [batch, 256]
        
        # Project back to weight space
        l_w = self.fc_out(lstm_out)  # [batch, 448] - predicted next weights
        
        # Gated update: w_t = f_t * w_{t-1} + (1 - f_t) * L(w_{t-1})
        w_t = f_t * w_t_minus_1 + (1 - f_t) * l_w
        
        return w_t, hidden
    
    def forward_no_gate(self, w_t_minus_1, hidden=None):
        """
        LSTM forward without fixation gating.
        Used during Stage 2 training where only boundary frames are processed.
        
        Args:
            w_t_minus_1: Current channel weights [batch, 448]
            hidden: LSTM hidden state or None
            
        Returns:
            l_w: Predicted next weights [batch, 448] (raw LSTM output, no gating)
            hidden: Updated hidden state
        """
        batch_size = w_t_minus_1.shape[0]
        
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=w_t_minus_1.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=w_t_minus_1.device)
            hidden = (h, c)
        
        x = self.fc_in(w_t_minus_1)
        x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.squeeze(1)
        l_w = self.fc_out(lstm_out)
        
        return l_w, hidden
    
    