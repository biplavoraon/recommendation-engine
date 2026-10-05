import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization


VALIDATION_DATA = "data/validation_data.pkl"

MODEL_PATH = "models/ranker_mf_model.pt"
INDEX_PATH = "models/ranker_item_index.faiss"
MAPPINGS_PATH = "models/ranker_mf_mappings.pkl"

NUM_CANDIDATES = 500


def main():

    with open(VALIDATION_DATA, "rb") as f:
        data = pickle.load(f)

    validation_targets = data["validation_targets"]
    history_items = data["validation_history_items"]

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = MatrixFactorization(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with open(MAPPINGS_PATH, "rb") as f:
        mappings = pickle.load(f)

    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    index = faiss.read_index(INDEX_PATH)

    ranks = []

    for user_id, target_item in validation_targets.items():

        if user_id not in user_to_idx:
            continue

        if target_item not in item_to_idx:
            continue

        user_idx = user_to_idx[user_id]

        with torch.no_grad():

            user_tensor = torch.tensor(
                [user_idx],
                dtype=torch.long,
            )

            user_embedding = (
                model.user_embedding(user_tensor)
                .numpy()
                .astype(np.float32)
            )

        seen = history_items.get(
            user_id,
            set(),
        )

        retrieve_k = min(
            index.ntotal,
            NUM_CANDIDATES + len(seen),
        )

        _, indices = index.search(
            user_embedding,
            retrieve_k,
        )

        candidates = []

        for item_idx in indices[0]:

            item_idx = int(item_idx)

            if item_idx < 0:
                continue

            item_id = idx_to_item[item_idx]

            if item_id in seen:
                continue

            candidates.append(item_id)

            if len(candidates) >= NUM_CANDIDATES:
                break

        if target_item not in candidates:
            continue

        rank = candidates.index(target_item) + 1

        ranks.append(rank)

    ranks = np.asarray(ranks)

    print()
    print("========================================")
    print("       CANDIDATE RANK ANALYSIS")
    print("========================================")
    print()

    print(f"Users with positive in candidates: {len(ranks)}")
    print()

    print(f"Mean rank:   {np.mean(ranks):.2f}")
    print(f"Median rank: {np.median(ranks):.2f}")
    print(f"Min rank:    {np.min(ranks)}")
    print(f"Max rank:    {np.max(ranks)}")
    print()

    print("Rank distribution")
    print("----------------------------------------")

    ranges = [
        ("1-10", 1, 10),
        ("11-50", 11, 50),
        ("51-100", 51, 100),
        ("101-250", 101, 250),
        ("251-500", 251, 500),
    ]

    for name, low, high in ranges:

        count = np.sum(
            (ranks >= low)
            & (ranks <= high)
        )

        percentage = (
            count / len(ranks) * 100
        )

        print(
            f"{name:10s}: "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )


if __name__ == "__main__":
    main()
