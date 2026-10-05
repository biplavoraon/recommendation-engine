import pickle

import torch

from app.evaluation import recall_at_k, ndcg_at_k
from app.ml.neural_collaborative_filtering import (
    NeuralCollaborativeFiltering,
)


K = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    # Load prepared data
    with open("data/mf_data.pkl", "rb") as f:
        data = pickle.load(f)

    train_items = data["train_items"]
    test_items = data["test_items"]
    user_to_idx = data["user_to_idx"]
    item_to_idx = data["item_to_idx"]

    # Load trained NCF model
    checkpoint = torch.load(
        "models/ncf_model.pt",
        map_location=DEVICE,
        weights_only=False,
    )

    model = NeuralCollaborativeFiltering(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Reverse mappings
    idx_to_user = {
        idx: user_id
        for user_id, idx in user_to_idx.items()
    }

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    recalls = []
    ndcgs = []

    with torch.no_grad():

        for user_idx, seen_items in train_items.items():

            # Original user ID
            user_id = idx_to_user[user_idx]

            # Held-out test item
            test_item_id = test_items[user_id]

            # User tensor
            user_tensor = torch.tensor(
                [user_idx],
                dtype=torch.long,
                device=DEVICE,
            )

            # Score every item using NCF
            scores = model.score_all_items(
                user_tensor
            )

            # Remove training interactions
            seen_items = list(seen_items)

            scores[seen_items] = float("-inf")

            # Top-K recommendations
            _, top_indices = torch.topk(
                scores,
                K,
            )

            recommendations = [
                idx_to_item[idx.item()]
                for idx in top_indices
            ]

            # One relevant test item
            relevant = [test_item_id]

            recalls.append(
                recall_at_k(
                    recommendations,
                    relevant,
                    K,
                )
            )

            ndcgs.append(
                ndcg_at_k(
                    recommendations,
                    relevant,
                    K,
                )
            )

    print(
        f"Users evaluated: {len(recalls)}"
    )

    print(
        f"Recall@{K}: "
        f"{sum(recalls) / len(recalls):.4f}"
    )

    print(
        f"NDCG@{K}:   "
        f"{sum(ndcgs) / len(ndcgs):.4f}"
    )


if __name__ == "__main__":
    main()
