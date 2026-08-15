import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x):
        sequence, _ = self.bilstm(x)
        return self.head(sequence[:, -1]).squeeze(-1)
