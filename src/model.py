import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder for network traffic anomaly detection.
    Trained only on benign traffic — high reconstruction error = anomaly.

    Architecture:
        Encoder: 42 → 32 → 16 → 8  (bottleneck)
        Decoder:  8 → 16 → 32 → 42
    """

    def __init__(self, input_dim: int = 42):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
            # No activation — output can be any real value (scaled data)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck representation (used for t-SNE later)."""
        return self.encoder(x)
