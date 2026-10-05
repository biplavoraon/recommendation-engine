import pickle

import faiss
import numpy as np
import torch


DATA_PATH = "data/mf_data.pkl"
MODEL_PATH = "models/mf_model.pt"
INDEX_PATH = "models/item_index_ivf.faiss"

NUM_CANDIDATES = 500
NPROBE = 16


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    index = faiss.read_index(INDEX_PATH)
    index.nprobe = NPROBE

    user_to_idx = data["user_to_idx"]
    train_items = data["train_items"]
    test_items = data["test_items"]
    item_to_idx = data["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    user_embeddings = checkpoint["model_state_dict"][
        "user_embedding.weight"
    ].numpy().astype(np.float32)

    hits = 0
    total = 0

    for user_id, user_idx in user_to_idx.items():
        seen_items = train_items[user_idx]
        test_item = test_items[user_id]

        # Retrieve extra items so we can filter seen items.
        search_k = min(
            NUM_CANDIDATES + len(seen_items),
            index.ntotal,
        )

        query = user_embeddings[user_idx].reshape(1, -1)

        _, indices = index.search(
            query,
            search_k,
        )

        candidates = []

        for item_idx in indices[0]:
            if item_idx == -1:
                continue

            item_id = idx_to_item[item_idx]

            if item_id in seen_items:
                continue

            candidates.append(item_id)

            if len(candidates) >= NUM_CANDIDATES:
                break

        if test_item in candidates:
            hits += 1

        total += 1

    recall = hits / total

    print(f"Users evaluated: {total}")
    print(f"Candidate count: {NUM_CANDIDATES}")
    print(f"nprobe: {NPROBE}")
    print(f"Hits: {hits}")
    print(f"Recall@{NUM_CANDIDATES}: {recall:.4f}")


if __name__ == "__main__":
    main()
