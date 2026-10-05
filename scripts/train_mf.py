import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.ml.dataset import BPRDataset
from app.ml.matrix_factorization import MatrixFactorization
from app.ml.losses import bpr_loss


SEED = 42
EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(SEED)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    with open(
        "data/mf_data.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)

    train_items = data["train_items"]

    num_users = len(data["user_to_idx"])
    num_items = len(data["item_to_idx"])

    dataset = BPRDataset(
        user_items=train_items,
        num_items=num_items,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Training samples: {len(dataset)}")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for users, positives, negatives in loader:

            # Move batch to GPU
            users = users.to(DEVICE)
            positives = positives.to(DEVICE)
            negatives = negatives.to(DEVICE)

            positive_scores = model(
                users,
                positives,
            )

            negative_scores = model(
                users,
                negatives,
            )

            loss = bpr_loss(
                positive_scores,
                negative_scores,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {average_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": EMBEDDING_DIM,
            "user_to_idx": data["user_to_idx"],
            "item_to_idx": data["item_to_idx"],
        },
        "models/mf_model.pt",
    )

    print(
        "Model saved to models/mf_model.pt"
    )


if __name__ == "__main__":
    main()
