import pickle

import numpy as np
import torch

from app.ml.ranker import NeuralRanker


DATA_PATH = "data/ranker_dataset_v2.pkl"
MODEL_PATH = "models/neural_ranker_v3.pt"

GROUP_SIZE = 100
TOP_K = 10


def recall_at_k(ranked_indices, k):
    return 1.0 if 0 in ranked_indices[:k] else 0.0


def ndcg_at_k(ranked_indices, k):
    for rank, idx in enumerate(ranked_indices[:k]):
        if idx == 0:
            return 1.0 / np.log2(rank + 2)
    return 0.0


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------------
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    X = np.asarray(data["X"], dtype=np.float32)

    num_groups = X.shape[0] // GROUP_SIZE

    print(f"Training examples: {X.shape[0]}")
    print(f"Training groups:   {num_groups}")
    print(f"Feature dimension: {X.shape[1]}")

    # ---------------------------------------------------------
    # Load ranker
    # ---------------------------------------------------------
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    ranker = NeuralRanker(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint.get("dropout", 0.2),
    )

    ranker.load_state_dict(
        checkpoint["model_state_dict"]
    )

    ranker.to(device)
    ranker.eval()

    # ---------------------------------------------------------
    # Evaluate groups
    # ---------------------------------------------------------
    recalls = []
    ndcgs = []

    with torch.no_grad():

        for group_idx in range(num_groups):

            start = group_idx * GROUP_SIZE
            end = start + GROUP_SIZE

            group = torch.tensor(
                X[start:end],
                dtype=torch.float32,
                device=device,
            )

            scores = (
                ranker(group)
                .cpu()
                .numpy()
            )

            ranking = np.argsort(-scores)

            recalls.append(
                recall_at_k(
                    ranking,
                    TOP_K,
                )
            )

            ndcgs.append(
                ndcg_at_k(
                    ranking,
                    TOP_K,
                )
            )

    print()
    print("========================================")
    print("       RANKER TRAINING EVALUATION")
    print("========================================")
    print()

    print(f"Groups evaluated: {num_groups}")
    print(f"Recall@{TOP_K}:       {np.mean(recalls):.4f}")
    print(f"NDCG@{TOP_K}:         {np.mean(ndcgs):.4f}")


if __name__ == "__main__":
    main()
