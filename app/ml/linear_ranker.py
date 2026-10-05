import torch
import torch.nn as nn


class LinearRanker(nn.Module):
    """
    Linear scoring model for candidate reranking.

    score(x) = w^T x + b
    """

    def __init__(self, input_dim):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            1,
        )

    def forward(self, x):
        return self.linear(x).squeeze(-1)
