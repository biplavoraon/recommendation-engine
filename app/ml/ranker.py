import torch
import torch.nn as nn


class NeuralRanker(nn.Module):
    def __init__(
        self,
        input_dim=141,
        hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
