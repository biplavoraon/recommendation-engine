import pickle

import torch

from app.candidate_generator import FAIISCandidateGenerator
from app.ml.matrix_factorization import MatrixFactorization


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

    user_id = 1
    user_idx = data["user_to_idx"][user_id]
    seen_items = data["train_items"][user_idx]

    candidates = generator.generate(
        user_idx=user_idx,
        seen_items=seen_items,
        num_candidates=100,
    )

    print(f"User: {user_id}")
    print(f"Seen items: {len(seen_items)}")
    print(f"Candidates generated: {len(candidates)}")

    print("\nFirst 10 candidates:")

    for rank, item_id in enumerate(candidates[:10], 1):
        print(f"{rank:2d}. item={item_id}")

    print(
        "\nSeen-item leakage:",
        bool(set(candidates) & set(seen_items)),
    )


if __name__ == "__main__":
    main()
