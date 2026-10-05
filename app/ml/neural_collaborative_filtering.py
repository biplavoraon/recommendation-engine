import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        hidden_dims=(128, 64),
        dropout=0.2,
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

        layers = []
        input_dim = 2 * embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, users, items):
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)

        x = torch.cat(
            [user_embeddings, item_embeddings],
            dim=-1,
        )

        return self.mlp(x).squeeze(-1)

    def score_all_items(self, user):
        user_embedding = self.user_embedding(user)

        item_indices = torch.arange(
            self.item_embedding.num_embeddings,
            device=user.device,
        )

        item_embeddings = self.item_embedding(item_indices)

        user_embeddings = user_embedding.expand(
            item_embeddings.size(0),
            -1,
        )

        x = torch.cat(
            [user_embeddings, item_embeddings],
            dim=-1,
        )

        return self.mlp(x).squeeze(-1)
