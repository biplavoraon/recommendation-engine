import pickle
import time

import faiss
import numpy as np
import torch


DATA_PATH = "data/mf_data.pkl"
MODEL_PATH = "models/mf_model.pt"
EXACT_INDEX_PATH = "models/item_index.faiss"
IVF_INDEX_PATH = "models/item_index_ivf.faiss"

K = 500
NPROBE = 16

WARMUP = 100
ITERATIONS = 1000


def benchmark(index, queries):
    # Warm-up
    for query in queries[:WARMUP]:
        index.search(query.reshape(1, -1), K)

    start = time.perf_counter()

    for query in queries:
        index.search(query.reshape(1, -1), K)

    elapsed = time.perf_counter() - start

    return elapsed / len(queries) * 1000


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    user_embeddings = checkpoint["model_state_dict"][
        "user_embedding.weight"
    ].numpy().astype(np.float32)

    # Repeat user embeddings so we have enough queries.
    queries = np.tile(
        user_embeddings,
        (
            (ITERATIONS // len(user_embeddings)) + 1,
            1,
        ),
    )[:ITERATIONS]

    exact_index = faiss.read_index(
        EXACT_INDEX_PATH
    )

    ivf_index = faiss.read_index(
        IVF_INDEX_PATH
    )

    ivf_index.nprobe = NPROBE

    exact_latency = benchmark(
        exact_index,
        queries,
    )

    ivf_latency = benchmark(
        ivf_index,
        queries,
    )

    speedup = exact_latency / ivf_latency

    print(f"Queries: {ITERATIONS}")
    print(f"Candidates: {K}")
    print()
    print(
        f"Exact IndexFlatIP: "
        f"{exact_latency:.4f} ms/query"
    )
    print(
        f"IVF nprobe={NPROBE}: "
        f"{ivf_latency:.4f} ms/query"
    )
    print(
        f"Speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
