import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization


MODEL_PATH = "models/ranker_mf_model.pt"
INDEX_PATH = "models/ranker_item_index.faiss"


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    embedding_dim = checkpoint["embedding_dim"]

    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Item embedding matrix
    item_embeddings = (
        model.item_embedding.weight
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    # Exact inner-product search
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embeddings)

    faiss.write_index(index, INDEX_PATH)

    # Save mappings separately
    with open(
        "models/ranker_mf_mappings.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "user_to_idx": checkpoint["user_to_idx"],
                "item_to_idx": checkpoint["item_to_idx"],
            },
            f,
        )

    print(f"Embedding dimension: {embedding_dim}")
    print(f"Items indexed: {index.ntotal}")
    print(f"Saved index to {INDEX_PATH}")
    print("Saved mappings to models/ranker_mf_mappings.pkl")


if __name__ == "__main__":
    main()
