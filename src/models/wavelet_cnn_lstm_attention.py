import torch
from torch import nn
import pywt
import numpy as np


class WaveletTransform(nn.Module):
    """
    Wavelet Transform Module for multi-scale feature extraction.
    Applies continuous wavelet transform to decompose input signals into
    different frequency components.
    """
    def __init__(self, wavelet='db4', levels=3):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            wavelet_features: (batch_size, seq_len, input_size * levels)
        """
        batch_size, seq_len, input_size = x.shape
        device = x.device
        
        # Process each sample in batch
        all_wavelet_features = []
        
        for b in range(batch_size):
            sample_wavelets = []
            
            # Process each feature
            for f in range(input_size):
                signal = x[b, :, f].cpu().detach().numpy()
                
                # Apply wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet, level=min(self.levels, int(np.log2(len(signal)))))
                
                # Normalize coefficients to match original signal length
                wavelet_decomp = []
                for coeff in coeffs:
                    if len(coeff) < seq_len:
                        coeff = np.pad(coeff, (0, seq_len - len(coeff)), mode='edge')
                    else:
                        coeff = coeff[:seq_len]
                    wavelet_decomp.append(coeff)
                
                wavelet_decomp = np.array(wavelet_decomp)  # (levels+1, seq_len)
                sample_wavelets.append(wavelet_decomp)
            
            # Stack all features: (input_size, levels+1, seq_len)
            sample_wavelets = np.array(sample_wavelets)
            # Reshape to (seq_len, input_size * (levels+1))
            sample_wavelets = sample_wavelets.transpose(2, 0, 1).reshape(seq_len, -1)
            all_wavelet_features.append(sample_wavelets)
        
        # Stack batch: (batch_size, seq_len, input_size * (levels+1))
        wavelet_features = np.array(all_wavelet_features)
        wavelet_features = torch.FloatTensor(wavelet_features).to(device)
        
        return wavelet_features


class Attention(nn.Module):
    """
    Attention mechanism for context-aware feature weighting.
    Computes attention weights across the sequence dimension.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_size)
        return context


class WaveletCNNLSTMAttentionRegressor(nn.Module):
    """
    Wavelet-Enhanced CNN-LSTM-Attention Model for Battery SOH Estimation.
    
    Architecture:
    1. Wavelet Transform: Multi-scale decomposition of input signal
    2. CNN: Local feature extraction from wavelet coefficients
    3. LSTM: Temporal dependency modeling
    4. Attention: Context-aware feature weighting
    5. Regression Head: SOH prediction
    
    This model is designed for accurate State-of-Health estimation in lithium-ion batteries
    by combining the advantages of:
    - Wavelet transforms for multi-scale pattern recognition
    - CNNs for efficient feature extraction
    - LSTMs for capturing long-term temporal dependencies
    - Attention mechanisms for context-aware predictions
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        wavelet='db4',
        wavelet_levels=3,
    ):
        super().__init__()
        
        # Wavelet Transform
        self.wavelet_transform = WaveletTransform(wavelet=wavelet, levels=wavelet_levels)
        wavelet_output_size = input_size * (wavelet_levels + 1)
        
        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(wavelet_output_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        
        # LSTM for temporal modeling
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        
        # Attention mechanism
        self.attention = Attention(hidden_size)
        
        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        Forward pass for SOH prediction.
        
        Args:
            x: (batch_size, seq_len, input_size) - Battery measurements
        
        Returns:
            soh_pred: (batch_size,) - Predicted SOH values
        """
        # Step 1: Wavelet Transform
        # Input: (batch_size, seq_len, input_size)
        x_wavelet = self.wavelet_transform(x)
        # Output: (batch_size, seq_len, input_size * (levels+1))
        
        # Step 2: CNN Feature Extraction
        # Transpose for Conv1d: (batch_size, channels, seq_len)
        x_cnn = x_wavelet.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        # Output: (batch_size, hidden_size, seq_len)
        
        # Step 3: Prepare for LSTM
        # Transpose back: (batch_size, seq_len, hidden_size)
        x_lstm = x_cnn.transpose(1, 2)
        
        # Step 4: LSTM Temporal Modeling
        lstm_out, _ = self.lstm(x_lstm)
        # Output: (batch_size, seq_len, hidden_size)
        
        # Step 5: Attention Mechanism
        context = self.attention(lstm_out)
        # Output: (batch_size, hidden_size)
        
        # Step 6: Regression Head
        soh_pred = self.head(context)
        # Output: (batch_size, 1)
        
        return soh_pred.squeeze(-1)
