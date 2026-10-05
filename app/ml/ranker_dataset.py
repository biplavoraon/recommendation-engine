import pickle

import torch
from torch.utils.data import Dataset


class RankerDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        X = torch.tensor(
            data["X"],
            dtype=torch.float32,
        )

        y = torch.tensor(
            data["y"],
            dtype=torch.float32,
        )

        # Read dataset structure from metadata.
        self.group_size = data["group_size"]
        self.feature_dim = data["feature_dim"]

        # Sanity checks.
        assert X.shape[1] == self.feature_dim

        assert len(X) % self.group_size == 0

        self.num_groups = len(X) // self.group_size

        # [num_groups * group_size, feature_dim]
        # ->
        # [num_groups, group_size, feature_dim]
        self.X = X.reshape(
            self.num_groups,
            self.group_size,
            self.feature_dim,
        )

        self.y = y.reshape(
            self.num_groups,
            self.group_size,
        )

    def __len__(self):
        return self.num_groups

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index],
        )
