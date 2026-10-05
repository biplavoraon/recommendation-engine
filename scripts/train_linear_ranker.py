import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.linear_ranker import LinearRanker
from app.ml.ranker_losses import pairwise_ranking_loss


DATA_PATH = "data/ranker_dataset_v4.pkl"
MODEL_PATH = "models/linear_ranker.pt"

INPUT_DIM = 13

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

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

    # ---------------------------------------------------------
    # Load V4 dataset
    # ---------------------------------------------------------
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X = np.asarray(
        data["X"],
        dtype=np.float32,
    )

    y = np.asarray(
        data["y"],
        dtype=np.float32,
    )

    group_size = data["group_size"]

    num_groups = len(X) // group_size

    print(f"Training groups:   {num_groups}")
    print(f"Group size:        {group_size}")
    print(f"Original features: {X.shape[1]}")

    # ---------------------------------------------------------
    # Extract:
    #
    # user features:  indices 128:134
    # item features:  indices 134:140
    # MF score:       index 140
    #
    # Ignore:
    # user embedding  0:64
    # item embedding 64:128
    # retrieval rank  141
    #
    # Total = 6 + 6 + 1 = 13
    # ---------------------------------------------------------
    X_linear = np.concatenate(
        [
            X[:, 128:134],
            X[:, 134:140],
            X[:, 140:141],
        ],
        axis=1,
    )

    assert X_linear.shape[1] == INPUT_DIM

    X_tensor = torch.tensor(
        X_linear,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Reshape into groups
    # ---------------------------------------------------------
    X_tensor = X_tensor.reshape(
        num_groups,
        group_size,
        INPUT_DIM,
    )

    y_tensor = y_tensor.reshape(
        num_groups,
        group_size,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = LinearRanker(
        input_dim=INPUT_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0
        batches = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            scores = model(
                X_batch.reshape(
                    -1,
                    INPUT_DIM,
                )
            )

            scores = scores.reshape(
                X_batch.shape[0],
                group_size,
            )

            positive_scores = scores[:, 0]

            negative_scores = scores[:, 1:]

            positive_scores = (
                positive_scores
                .unsqueeze(1)
                .expand_as(negative_scores)
                .reshape(-1)
            )

            negative_scores = (
                negative_scores.reshape(-1)
            )

            loss = pairwise_ranking_loss(
                positive_scores,
                negative_scores,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
        },
        MODEL_PATH,
    )

    print(
        f"Saved linear ranker to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
