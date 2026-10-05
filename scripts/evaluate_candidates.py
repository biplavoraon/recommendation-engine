import pickle

import torch

from app.candidate_generator import FAIISCandidateGenerator
from app.ml.matrix_factorization import MatrixFactorization


MODEL_PATH = "models/mf_model.pt"
DATA_PATH = "data/mf_data.pkl"
INDEX_PATH = "models/item_index.faiss"

NUM_CANDIDATES = 500


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

    generator = FAIISCandidateGenerator(
        model=model,
        index_path=INDEX_PATH,
        item_to_idx=data["item_to_idx"],
    )

    user_to_idx = data["user_to_idx"]
    train_items = data["train_items"]
    test_items = data["test_items"]

    hits = 0
    total = 0

    for user_id, user_idx in user_to_idx.items():
        seen_items = train_items[user_idx]
        test_item = test_items[user_id]

        candidates = generator.generate(
            user_idx=user_idx,
            seen_items=seen_items,
            num_candidates=NUM_CANDIDATES,
        )

        if test_item in candidates:
            hits += 1

        total += 1

    recall = hits / total

    print(f"Users evaluated: {total}")
    print(f"Candidate count: {NUM_CANDIDATES}")
    print(f"Hits: {hits}")
    print(f"Recall@{NUM_CANDIDATES}: {recall:.4f}")


if __name__ == "__main__":
    main()
