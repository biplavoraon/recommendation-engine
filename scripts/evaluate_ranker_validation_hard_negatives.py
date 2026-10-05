import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization
from app.ml.ranker import NeuralRanker
from app.evaluation import recall_at_k, ndcg_at_k


VALIDATION_DATA = "data/validation_data.pkl"
VALIDATION_FEATURES = "data/validation_features.pkl"

MF_MODEL = "models/ranker_mf_model.pt"
FAISS_INDEX = "models/ranker_item_index.faiss"
RANKER_MODEL = "models/neural_ranker.pt"
MAPPINGS = "models/ranker_mf_mappings.pkl"

NUM_CANDIDATES = 500
NUM_NEGATIVES = 20
K = 10


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------
    with open(VALIDATION_DATA, "rb") as f:
        validation = pickle.load(f)

    validation_targets = validation["validation_targets"]
    validation_history_items = validation[
        "validation_history_items"
    ]

    # ---------------------------------------------------------
    # Load validation features
    # ---------------------------------------------------------
    with open(VALIDATION_FEATURES, "rb") as f:
        features = pickle.load(f)

    user_stats = features["user_stats"].set_index("user_id")
    item_stats = features["item_stats"].set_index("item_id")

    user_feature_columns = [
        "user_interaction_count",
        "user_avg_rating",
        "user_rating_std",
        "user_unique_items",
        "user_days_since_last_interaction",
        "user_active_days",
    ]

    item_feature_columns = [
        "item_interaction_count",
        "item_avg_rating",
        "item_rating_std",
        "item_unique_users",
        "item_log_popularity",
        "item_days_since_last_interaction",
    ]

    # ---------------------------------------------------------
    # Load mappings
    # ---------------------------------------------------------
    with open(MAPPINGS, "rb") as f:
        mappings = pickle.load(f)

    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    # ---------------------------------------------------------
    # Load MF
    # ---------------------------------------------------------
    checkpoint = torch.load(
        MF_MODEL,
        map_location=device,
        weights_only=False,
    )

    mf = MatrixFactorization(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )

    mf.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mf.to(device)
    mf.eval()

    # ---------------------------------------------------------
    # Load FAISS
    # ---------------------------------------------------------
    index = faiss.read_index(FAISS_INDEX)

    # ---------------------------------------------------------
    # Load neural ranker
    # ---------------------------------------------------------
    ranker_checkpoint = torch.load(
        RANKER_MODEL,
        map_location=device,
        weights_only=False,
    )

    ranker = NeuralRanker(
        input_dim=ranker_checkpoint["input_dim"],
        hidden_dim=ranker_checkpoint["hidden_dim"],
        dropout=ranker_checkpoint.get("dropout", 0.2),
    )

    ranker.load_state_dict(
        ranker_checkpoint["model_state_dict"]
    )

    ranker.to(device)
    ranker.eval()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    mf_recalls = []
    mf_ndcgs = []

    ranker_recalls = []
    ranker_ndcgs = []

    candidate_hits = 0
    users_evaluated = 0

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    with torch.no_grad():

        for user_id, positive_item_id in validation_targets.items():

            user_id = int(user_id)
            positive_item_id = int(positive_item_id)

            if user_id not in user_to_idx:
                continue

            if positive_item_id not in item_to_idx:
                continue

            # -------------------------------------------------
            # User embedding
            # -------------------------------------------------
            user_idx = user_to_idx[user_id]

            user_tensor = torch.tensor(
                [user_idx],
                dtype=torch.long,
                device=device,
            )

            user_embedding = (
                mf.user_embedding(user_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # User features
            # -------------------------------------------------
            user_features = (
                user_stats.loc[
                    user_id,
                    user_feature_columns,
                ]
                .to_numpy(dtype=np.float32)
            )

            # -------------------------------------------------
            # Retrieve 500 candidates
            # -------------------------------------------------
            history_items = validation_history_items.get(
                user_id,
                set(),
            )

            retrieve_k = min(
                index.ntotal,
                NUM_CANDIDATES + len(history_items),
            )

            scores, indices = index.search(
                user_embedding.reshape(1, -1),
                retrieve_k,
            )

            candidates = []

            for score, item_idx in zip(
                scores[0],
                indices[0],
            ):

                item_idx = int(item_idx)

                if item_idx < 0:
                    continue

                item_id = idx_to_item[item_idx]

                if item_id in history_items:
                    continue

                candidates.append(
                    (item_id, float(score))
                )

                if len(candidates) >= NUM_CANDIDATES:
                    break

            # -------------------------------------------------
            # Positive must be in candidates
            # -------------------------------------------------
            positive_position = None

            for position, (item_id, _) in enumerate(candidates):
                if item_id == positive_item_id:
                    positive_position = position
                    break

            if positive_position is None:
                continue

            candidate_hits += 1

            # -------------------------------------------------
            # Select exactly the same type of negatives
            # as training.
            # -------------------------------------------------
            negative_candidates = [
                candidate
                for candidate in candidates
                if candidate[0] != positive_item_id
            ]

            negative_candidates = negative_candidates[
                :NUM_NEGATIVES
            ]

            # If fewer than 20 are available, skip.
            if len(negative_candidates) < NUM_NEGATIVES:
                continue

            # -------------------------------------------------
            # Positive + 20 hard negatives
            # -------------------------------------------------
            selected_candidates = [
                (positive_item_id, None)
            ] + negative_candidates

            # -------------------------------------------------
            # MF baseline on same 21 candidates
            # -------------------------------------------------
            mf_candidate_ids = [
                item_id
                for item_id, _ in selected_candidates
            ]

            mf_candidate_scores = []

            for item_id in mf_candidate_ids:

                item_idx = item_to_idx[item_id]

                item_tensor = torch.tensor(
                    [item_idx],
                    dtype=torch.long,
                    device=device,
                )

                item_embedding = (
                    mf.item_embedding(item_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                score = np.dot(
                    user_embedding,
                    item_embedding,
                )

                mf_candidate_scores.append(score)

            mf_order = np.argsort(
                -np.asarray(mf_candidate_scores)
            )

            mf_ranked = [
                mf_candidate_ids[i]
                for i in mf_order
            ]

            mf_recalls.append(
                recall_at_k(
                    mf_ranked,
                    [positive_item_id],
                    K,
                )
            )

            mf_ndcgs.append(
                ndcg_at_k(
                    mf_ranked,
                    [positive_item_id],
                    K,
                )
            )

            # -------------------------------------------------
            # Neural ranker features
            # -------------------------------------------------
            feature_rows = []

            for item_id, _ in selected_candidates:

                item_idx = item_to_idx[item_id]

                item_tensor = torch.tensor(
                    [item_idx],
                    dtype=torch.long,
                    device=device,
                )

                item_embedding = (
                    mf.item_embedding(item_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                if item_id in item_stats.index:
                    item_features = (
                        item_stats.loc[
                            item_id,
                            item_feature_columns,
                        ]
                        .to_numpy(dtype=np.float32)
                    )
                else:
                    item_features = np.zeros(
                        6,
                        dtype=np.float32,
                    )

                mf_score = np.float32(
                    np.dot(
                        user_embedding,
                        item_embedding,
                    )
                )

                feature_vector = np.concatenate(
                    [
                        user_embedding,
                        item_embedding,
                        user_features,
                        item_features,
                        np.asarray(
                            [mf_score],
                            dtype=np.float32,
                        ),
                    ]
                )

                assert feature_vector.shape == (141,)

                feature_rows.append(feature_vector)

            X = torch.tensor(
                np.asarray(feature_rows),
                dtype=torch.float32,
                device=device,
            )

            # -------------------------------------------------
            # Neural ranker
            # -------------------------------------------------
            ranker_scores = (
                ranker(X)
                .cpu()
                .numpy()
            )

            ranker_order = np.argsort(
                -ranker_scores
            )

            ranker_ranked = [
                selected_candidates[i][0]
                for i in ranker_order
            ]

            ranker_recalls.append(
                recall_at_k(
                    ranker_ranked,
                    [positive_item_id],
                    K,
                )
            )

            ranker_ndcgs.append(
                ndcg_at_k(
                    ranker_ranked,
                    [positive_item_id],
                    K,
                )
            )

            users_evaluated += 1

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    candidate_recall = (
        candidate_hits
        / len(validation_targets)
    )

    print()
    print("========================================")
    print("  HARD-NEGATIVE VALIDATION")
    print("========================================")
    print()

    print(
        f"Validation users:      "
        f"{len(validation_targets)}"
    )

    print(
        f"Users evaluated:       "
        f"{users_evaluated}"
    )

    print(
        f"Candidate Recall@500:  "
        f"{candidate_recall:.4f}"
    )

    print()

    print("----------------------------------------")
    print("MF / FAISS")
    print("----------------------------------------")

    print(
        f"Recall@10:             "
        f"{np.mean(mf_recalls):.4f}"
    )

    print(
        f"NDCG@10:               "
        f"{np.mean(mf_ndcgs):.4f}"
    )

    print()

    print("----------------------------------------")
    print("Neural Ranker")
    print("----------------------------------------")

    print(
        f"Recall@10:             "
        f"{np.mean(ranker_recalls):.4f}"
    )

    print(
        f"NDCG@10:               "
        f"{np.mean(ranker_ndcgs):.4f}"
    )

    print()


if __name__ == "__main__":
    main()
