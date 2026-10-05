import pickle

import faiss
import numpy as np
import torch


MODEL_PATH = "models/mf_model.pt"
DATA_PATH = "data/mf_data.pkl"
INDEX_PATH = "models/item_index.faiss"


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    item_embeddings = checkpoint["model_state_dict"][
        "item_embedding.weight"
    ].numpy()

    embedding_dim = item_embeddings.shape[1]

    # FAISS expects float32 vectors.
    item_embeddings = np.ascontiguousarray(
        item_embeddings.astype(np.float32)
    )

    # Inner product = dot product.
    index = faiss.IndexFlatIP(embedding_dim)

    index.add(item_embeddings)

    print(f"Embedding dimension: {embedding_dim}")
    print(f"Items indexed: {index.ntotal}")

    faiss.write_index(index, INDEX_PATH)

    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
