import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_size)
        return context


class CNNBiLSTMAttentionRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=True,
        )
        
        self.attention = Attention(hidden_size * 2)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.cnn(x)  # (batch_size, hidden_size, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        
        sequence, _ = self.bilstm(x)  # (batch_size, seq_len, hidden_size*2)
        context = self.attention(sequence)  # (batch_size, hidden_size*2)
        
        return self.head(context).squeeze(-1)
