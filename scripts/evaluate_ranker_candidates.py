import pickle

import faiss
import torch

from app.ml.matrix_factorization import MatrixFactorization


EXAMPLES_PATH = "data/ranker_examples.pkl"
MODEL_PATH = "models/ranker_mf_model.pt"
INDEX_PATH = "models/ranker_item_index.faiss"

K = 500


def main():
    with open(EXAMPLES_PATH, "rb") as f:
        examples = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    index = faiss.read_index(INDEX_PATH)

    model = MatrixFactorization(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    user_to_idx = checkpoint["user_to_idx"]
    item_to_idx = checkpoint["item_to_idx"]

    # Reverse mapping
    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    train_history = examples["train_history_items"]
    positives = examples["train_positive"]

    hits = 0
    evaluated = 0

    for user_id, positive_item_id in positives.items():

        if user_id not in user_to_idx:
            print(
                f"Skipping user {user_id}: "
                "missing user mapping"
            )
            continue

        if positive_item_id not in item_to_idx:
            print(
                f"Skipping user {user_id}: "
                f"missing item {positive_item_id}"
            )
            continue

        user_idx = user_to_idx[user_id]

        user_tensor = torch.tensor(
            [user_idx],
            dtype=torch.long,
        )

        with torch.no_grad():
            user_embedding = (
                model.user_embedding(user_tensor)
                .numpy()
                .astype("float32")
            )

        # Retrieve enough items to compensate for
        # history items that will be filtered out.
        retrieve_k = min(
            index.ntotal,
            K + len(train_history[user_id]),
        )

        _, indices = index.search(
            user_embedding,
            retrieve_k,
        )

        candidates = []

        seen_items = train_history[user_id]

        for item_idx in indices[0]:
            item_idx = int(item_idx)

            if item_idx < 0:
                continue

            item_id = idx_to_item[item_idx]

            # Production-style filtering:
            # don't recommend something already seen.
            if item_id in seen_items:
                continue

            candidates.append(item_id)

            if len(candidates) >= K:
                break

        if positive_item_id in candidates:
            hits += 1

        evaluated += 1

    recall = hits / evaluated if evaluated else 0.0

    print("=== Ranker Candidate Evaluation ===")
    print(f"Users evaluated: {evaluated}")
    print(f"Hits@{K}:         {hits}")
    print(f"Recall@{K}:       {recall:.4f}")


if __name__ == "__main__":
    main()
