import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.ml.dataset import BPRDataset
from app.ml.losses import bpr_loss
from app.ml.matrix_factorization import MatrixFactorization


INPUT_PATH = "data/validation_data.pkl"
OUTPUT_PATH = "models/validation_mf_model.pt"

EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)

    history_items = data["validation_history_items"]

    users = sorted(history_items.keys())

    # Full catalog from the existing ranker MF mapping
    with open(
        "models/ranker_mf_mappings.pkl",
        "rb",
    ) as f:
        mappings = pickle.load(f)

    item_to_idx = mappings["item_to_idx"]

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(users)
    }

    indexed_history = {
        user_to_idx[user_id]: {
            item_to_idx[item_id]
            for item_id in items
            if item_id in item_to_idx
        }
        for user_id, items in history_items.items()
    }

    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(
        f"Training interactions: "
        f"{sum(len(x) for x in indexed_history.values())}"
    )

    dataset = BPRDataset(
        indexed_history,
        num_items,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        batches = 0

        for users_batch, positives, negatives in loader:
            users_batch = users_batch.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            positive_scores = model(
                users_batch,
                positives,
            )

            negative_scores = model(
                users_batch,
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
            batches += 1

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {total_loss / batches:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": EMBEDDING_DIM,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
        },
        OUTPUT_PATH,
    )

    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
