import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        self.user_bias = nn.Embedding(
            num_users,
            1,
        )

        self.item_bias = nn.Embedding(
            num_items,
            1,
        )

        self.global_bias = nn.Parameter(
            torch.zeros(1)
        )

        self._initialize()

    def _initialize(self):

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.01,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            std=0.01,
        )

        nn.init.zeros_(
            self.user_bias.weight
        )

        nn.init.zeros_(
            self.item_bias.weight
        )

    def forward(self, users, items):

        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        interaction = (
            user_emb * item_emb
        ).sum(dim=1)

        user_bias = self.user_bias(users).squeeze(-1)
        item_bias = self.item_bias(items).squeeze(-1)

        return (
            interaction
            + user_bias
            + item_bias
            + self.global_bias
        )

    def score_all_items(self, user):
        user_emb = self.user_embedding(user)

        scores = user_emb @ self.item_embedding.weight.T

        scores += self.item_bias.weight.squeeze(-1)
        scores += self.user_bias(user).squeeze(-1)
        scores += self.global_bias

        return scores.squeeze(0)
