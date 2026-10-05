import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.ranker import NeuralRanker
from app.ml.ranker_losses import pairwise_ranking_loss


DATA_PATH = "data/ranker_dataset_v2.pkl"
OUTPUT_PATH = "models/neural_ranker_v2.pt"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

GROUP_SIZE = 100


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X = torch.tensor(
        data["X"],
        dtype=torch.float32,
    )

    y = torch.tensor(
        data["y"],
        dtype=torch.float32,
    )

    feature_dim = data["feature_dim"]

    num_groups = len(X) // GROUP_SIZE

    X = X.reshape(
        num_groups,
        GROUP_SIZE,
        feature_dim,
    )

    y = y.reshape(
        num_groups,
        GROUP_SIZE,
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"Training groups:   {num_groups}")
    print(f"Group size:        {GROUP_SIZE}")
    print(f"Feature dimension: {feature_dim}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = NeuralRanker(
        input_dim=feature_dim,
        hidden_dim=128,
        dropout=0.2,
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        batches = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            scores = model(X_batch)

            positive_scores = scores[:, 0]

            negative_scores = scores[:, 1:]

            positive_scores = (
                positive_scores
                .unsqueeze(1)
                .expand_as(negative_scores)
            )

            loss = pairwise_ranking_loss(
                positive_scores.reshape(-1),
                negative_scores.reshape(-1),
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
    # Save checkpoint
    # ---------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": feature_dim,
            "hidden_dim": 128,
            "dropout": 0.2,
            "group_size": GROUP_SIZE,
        },
        OUTPUT_PATH,
    )

    print(f"Saved ranker to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
