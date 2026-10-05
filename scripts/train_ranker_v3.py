import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.ranker import NeuralRanker
from app.ml.ranker_losses import listwise_ranking_loss


DATA_PATH = "data/ranker_dataset_v2.pkl"
OUTPUT_PATH = "models/neural_ranker_v3.pt"

BATCH_SIZE = 32
EPOCHS = 50
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

    feature_dim = data["feature_dim"]

    num_groups = len(X) // GROUP_SIZE

    X = X.reshape(
        num_groups,
        GROUP_SIZE,
        feature_dim,
    )

    dataset = TensorDataset(X)

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

        for (X_batch,) in loader:

            X_batch = X_batch.to(device)

            # -------------------------------------------------
            # Score every candidate in the group
            # -------------------------------------------------
            batch_size = X_batch.shape[0]

            scores = model(
                X_batch.reshape(
                    batch_size * GROUP_SIZE,
                    feature_dim,
                )
            )

            scores = scores.reshape(
                batch_size,
                GROUP_SIZE,
            )

            # -------------------------------------------------
            # Listwise loss
            #
            # Candidate 0 is the positive item.
            # -------------------------------------------------
            loss = listwise_ranking_loss(
                scores
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
            "input_dim": feature_dim,
            "hidden_dim": 128,
            "dropout": 0.2,
            "group_size": GROUP_SIZE,
            "loss": "listwise_softmax",
        },
        OUTPUT_PATH,
    )

    print(
        f"Saved ranker to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
