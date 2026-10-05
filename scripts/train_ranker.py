import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.ml.ranker import NeuralRanker
from app.ml.ranker_dataset import RankerDataset
from app.ml.ranker_losses import pairwise_ranking_loss


DATA_PATH = "data/ranker_dataset.pkl"
MODEL_PATH = "models/neural_ranker.pt"

INPUT_DIM = 141
HIDDEN_DIM = 128

BATCH_SIZE = 64
EPOCHS = 30
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

    dataset = RankerDataset(DATA_PATH)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"Training groups: {len(dataset)}")
    print(f"Group size:      21")
    print(f"Feature dimension:{INPUT_DIM}")

    model = NeuralRanker(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
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

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            # -------------------------------------------------
            # Flatten candidates
            # -------------------------------------------------
            batch_size = X.shape[0]

            X_flat = X.reshape(
                -1,
                INPUT_DIM,
            )

            scores = model(X_flat)

            scores = scores.reshape(
                batch_size,
                21,
            )

            # -------------------------------------------------
            # Positive is always index 0
            # -------------------------------------------------
            positive_scores = scores[:, 0]

            # Remaining 20 are negatives
            negative_scores = scores[:, 1:]

            # Compare each positive against every negative.
            positive_scores = (
                positive_scores
                .unsqueeze(1)
                .expand_as(negative_scores)
                .reshape(-1)
            )

            negative_scores = (
                negative_scores
                .reshape(-1)
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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
        },
        MODEL_PATH,
    )

    print(
        f"Saved ranker to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
