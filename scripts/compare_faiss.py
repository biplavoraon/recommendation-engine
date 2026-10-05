import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization


MODEL_PATH = "models/mf_model.pt"
DATA_PATH = "data/mf_data.pkl"
INDEX_PATH = "models/item_index.faiss"

K = 10


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = MatrixFactorization(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    index = faiss.read_index(INDEX_PATH)

    user_to_idx = data["user_to_idx"]
    item_to_idx = data["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    user_id = next(iter(user_to_idx))
    user_idx = user_to_idx[user_id]

    # --------------------------------------------------
    # PyTorch exact retrieval
    # --------------------------------------------------

    with torch.no_grad():
        user_tensor = torch.tensor(
            [user_idx],
            dtype=torch.long,
        )

        exact_scores = model.score_all_items(user_tensor)

        _, exact_indices = torch.topk(
            exact_scores,
            K,
        )

    exact_items = [
        idx_to_item[idx.item()]
        for idx in exact_indices
    ]

    # --------------------------------------------------
    # FAISS retrieval
    # --------------------------------------------------

    user_embedding = (
        model.user_embedding.weight[user_idx]
        .detach()
        .numpy()
        .astype(np.float32)
    )

    query = user_embedding.reshape(1, -1)

    faiss_scores, faiss_indices = index.search(
        query,
        K,
    )

    faiss_items = [
        idx_to_item[idx]
        for idx in faiss_indices[0]
    ]

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    print(f"User: {user_id}")

    print("\nPyTorch exact:")
    for rank, item in enumerate(exact_items, 1):
        print(f"{rank:2d}. item={item}")

    print("\nFAISS:")
    for rank, item in enumerate(faiss_items, 1):
        print(f"{rank:2d}. item={item}")

    print("\nSame ranking:", exact_items == faiss_items)
    print(
        "Same items:",
        set(exact_items) == set(faiss_items),
    )


if __name__ == "__main__":
    main()
