"""
Regularized Multi-Output Sequence Model for Agra Air Quality Forecasting.
Includes input noise jitter, sequence dropout, and compact parameter capacity to prevent overfitting.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LSTMHyperparameters


class TemporalAttention(nn.Module):
    """Computes attention weights over the 24-hour lookback sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq_outputs: (Batch, Seq_Len, Hidden_Dim)
        attn_scores = self.attn(seq_outputs)  # (Batch, Seq_Len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (Batch, Seq_Len, 1)
        context_vector = torch.sum(attn_weights * seq_outputs, dim=1)  # (Batch, Hidden_Dim)
        return context_vector, attn_weights


class AgraMultiOutputLSTM(nn.Module):
    """
    Regularized Deep Sequence Model designed to prevent overfitting on atmospheric time series:
    - Input Jitter (Gaussian noise during training)
    - Input Feature Dropout
    - Compact GRU/LSTM recurrent cell (64 units)
    - LayerNorm + Multi-Head Dropout (0.30)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.30,
        bidirectional: bool = False,
        cell_type: str = "gru",
        noise_std: float = 0.02
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.noise_std = noise_std
        self.dropout_rate = dropout

        # 1. Anti-overfitting: Input Feature Dropout
        self.input_drop = nn.Dropout(0.15)

        # 2. Recurrent backbone (GRU / LSTM)
        if self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )

        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 3. Attention pooling over 24 hours
        self.attention = TemporalAttention(rnn_out_dim)

        # 4. Regularized shared feature representation
        self.shared_fc = nn.Sequential(
            nn.Linear(rnn_out_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. Dedicated multi-pollutant prediction heads with LayerNorm
        self.head_pm25 = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(24, 1)
        )
        self.head_pm10 = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(24, 1)
        )
        self.head_ozone = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(24, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 24, Input_Dim)
        if self.training and self.noise_std > 0:
            # Data augmentation: Add slight Gaussian jitter to prevent memorization
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        x = self.input_drop(x)

        if self.cell_type == "gru":
            rnn_out, _ = self.rnn(x)
        else:
            rnn_out, _ = self.rnn(x)

        last_hidden = rnn_out[:, -1, :]
        context_vector, _ = self.attention(rnn_out)

        combined = torch.cat([last_hidden, context_vector], dim=1)
        feat = self.shared_fc(combined)

        out_pm25 = self.head_pm25(feat)
        out_pm10 = self.head_pm10(feat)
        out_ozone = self.head_ozone(feat)

        return torch.cat([out_pm25, out_pm10, out_ozone], dim=1)
