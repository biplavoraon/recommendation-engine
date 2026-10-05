import pickle

import faiss
import numpy as np
import torch


MODEL_PATH = "models/mf_model.pt"
DATA_PATH = "data/mf_data.pkl"
INDEX_PATH = "models/item_index_ivf.faiss"

N_LIST = 32


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

    item_embeddings = np.ascontiguousarray(
        item_embeddings.astype(np.float32)
    )

    embedding_dim = item_embeddings.shape[1]

    # Quantizer used to find the nearest IVF clusters.
    quantizer = faiss.IndexFlatIP(embedding_dim)

    # IVF index.
    index = faiss.IndexIVFFlat(
        quantizer,
        embedding_dim,
        N_LIST,
        faiss.METRIC_INNER_PRODUCT,
    )

    # IVF must be trained before adding vectors.
    print("Training IVF index...")
    index.train(item_embeddings)

    print("Adding item embeddings...")
    index.add(item_embeddings)

    faiss.write_index(index, INDEX_PATH)

    print(f"Embedding dimension: {embedding_dim}")
    print(f"Items indexed: {index.ntotal}")
    print(f"Number of clusters: {N_LIST}")
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
