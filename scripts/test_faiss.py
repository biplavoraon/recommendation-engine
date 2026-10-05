import pickle

import faiss
import torch


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

    index = faiss.read_index(INDEX_PATH)

    user_to_idx = data["user_to_idx"]
    item_to_idx = data["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    # Pick one user.
    user_id = next(iter(user_to_idx))
    user_idx = user_to_idx[user_id]

    user_embedding = checkpoint["model_state_dict"][
        "user_embedding.weight"
    ][user_idx].numpy()

    # FAISS expects shape: (number_of_queries, embedding_dim)
    query = user_embedding.reshape(1, -1).astype("float32")

    scores, indices = index.search(query, K)

    print(f"User: {user_id}")
    print(f"Top {K} recommendations:")

    for rank, (item_idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        item_id = idx_to_item[item_idx]

        print(
            f"{rank:2d}. item={item_id:4d} "
            f"score={score:.4f}"
        )


if __name__ == "__main__":
    main()
