import random

import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """
    Dataset for Bayesian Personalized Ranking (BPR).

    Each example consists of:
        user
        positive item
        negative item

    Positive items come from observed interactions.
    Negative items are sampled from items the user
    has not interacted with.
    """

    def __init__(
        self,
        interactions,
        num_items,
        num_negatives=1,
        seed=42,
    ):
        self.interactions = interactions
        self.num_items = num_items
        self.num_negatives = num_negatives

        self.rng = random.Random(seed)

        # Build the set of items each user has interacted with.
        self.user_items = {}

        for user_id, item_id in interactions:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()

            self.user_items[user_id].add(item_id)

        # One dataset example for every positive interaction.
        self.examples = list(interactions)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        user_id, positive_item_id = self.examples[index]

        negative_items = []

        interacted_items = self.user_items[user_id]

        while len(negative_items) < self.num_negatives:

            negative_item_id = self.rng.randrange(
                self.num_items
            )

            if negative_item_id in interacted_items:
                continue

            negative_items.append(negative_item_id)

        return (
            torch.tensor(
                user_id,
                dtype=torch.long,
            ),
            torch.tensor(
                positive_item_id,
                dtype=torch.long,
            ),
            torch.tensor(
                negative_items,
                dtype=torch.long,
            ),
        )
