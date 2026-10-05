import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization
from app.ml.ranker import NeuralRanker
from app.evaluation import recall_at_k, ndcg_at_k


RANKER_MF_MODEL = "models/ranker_mf_model.pt"
FAISS_INDEX = "models/ranker_item_index.faiss"
RANKER_MAPPINGS = "models/ranker_mf_mappings.pkl"
RANKER_MODEL = "models/neural_ranker_v4.pt"

VALIDATION_DATA = "data/validation_data.pkl"
VALIDATION_FEATURES = "data/validation_features.pkl"

K_CANDIDATES = 500
K_EVAL = 10


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------
    with open(VALIDATION_DATA, "rb") as f:
        validation_data = pickle.load(f)

    validation_targets = validation_data["validation_targets"]

    validation_history_items = validation_data[
        "validation_history_items"
    ]

    # ---------------------------------------------------------
    # Load validation features
    # ---------------------------------------------------------
    with open(VALIDATION_FEATURES, "rb") as f:
        validation_features = pickle.load(f)

    user_stats = validation_features["user_stats"]
    item_stats = validation_features["item_stats"]

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
    # Convert feature DataFrames into dictionaries
    # ---------------------------------------------------------
    user_feature_dict = {}

    for _, row in user_stats.iterrows():

        user_id = int(row["user_id"])

        user_feature_dict[user_id] = np.asarray(
            [
                row[col]
                for col in user_feature_columns
            ],
            dtype=np.float32,
        )

    item_feature_dict = {}

    for _, row in item_stats.iterrows():

        item_id = int(row["item_id"])

        item_feature_dict[item_id] = np.asarray(
            [
                row[col]
                for col in item_feature_columns
            ],
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Load MF mappings
    # ---------------------------------------------------------
    with open(RANKER_MAPPINGS, "rb") as f:
        mappings = pickle.load(f)

    user_to_idx = mappings["user_to_idx"]
    item_to_idx = mappings["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")

    # ---------------------------------------------------------
    # Load fixed MF model
    # ---------------------------------------------------------
    checkpoint = torch.load(
        RANKER_MF_MODEL,
        map_location=device,
        weights_only=False,
    )

    embedding_dim = checkpoint["embedding_dim"]

    mf_model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
    )

    mf_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    mf_model.to(device)
    mf_model.eval()

    # ---------------------------------------------------------
    # Load FAISS index
    # ---------------------------------------------------------
    index = faiss.read_index(FAISS_INDEX)

    # ---------------------------------------------------------
    # Load V4 neural ranker
    # ---------------------------------------------------------
    ranker_checkpoint = torch.load(
        RANKER_MODEL,
        map_location=device,
        weights_only=False,
    )

    ranker = NeuralRanker(
        input_dim=ranker_checkpoint["input_dim"],
        hidden_dim=ranker_checkpoint["hidden_dim"],
        dropout=0.2,
    )

    ranker.load_state_dict(
        ranker_checkpoint["model_state_dict"]
    )

    ranker.to(device)
    ranker.eval()

    # ---------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------
    assert ranker_checkpoint["input_dim"] == 142

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
    # Evaluation
    # ---------------------------------------------------------
    with torch.no_grad():

        for user_id, target_item in validation_targets.items():

            user_id = int(user_id)
            target_item = int(target_item)

            if user_id not in user_to_idx:
                continue

            if target_item not in item_to_idx:
                continue

            # -------------------------------------------------
            # User features
            # -------------------------------------------------
            if user_id not in user_feature_dict:
                continue

            user_feat = user_feature_dict[user_id]

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
                mf_model.user_embedding(user_tensor)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # Retrieve 500 candidates
            # -------------------------------------------------
            seen_items = validation_history_items.get(
                user_id,
                set(),
            )

            search_k = min(
                K_CANDIDATES + len(seen_items),
                num_items,
            )

            query = user_embedding.reshape(
                1,
                -1,
            ).astype(np.float32)

            faiss_scores, candidate_indices = index.search(
                query,
                search_k,
            )

            candidate_indices = candidate_indices[0]

            candidates = []
            candidate_scores = []

            for score, item_idx in zip(
                faiss_scores[0],
                candidate_indices,
            ):

                item_idx = int(item_idx)

                if item_idx not in idx_to_item:
                    continue

                item_id = idx_to_item[item_idx]

                # Remove historical items.
                if item_id in seen_items:
                    continue

                candidates.append(item_id)
                candidate_scores.append(
                    float(score)
                )

                if len(candidates) >= K_CANDIDATES:
                    break

            # -------------------------------------------------
            # Candidate recall
            # -------------------------------------------------
            if target_item not in candidates:
                continue

            candidate_hits += 1
            users_evaluated += 1

            # -------------------------------------------------
            # MF / FAISS baseline
            #
            # candidates are already ordered by FAISS score.
            # -------------------------------------------------
            mf_top10 = candidates[:K_EVAL]

            mf_recalls.append(
                recall_at_k(
                    mf_top10,
                    [target_item],
                    K_EVAL,
                )
            )

            mf_ndcgs.append(
                ndcg_at_k(
                    mf_top10,
                    [target_item],
                    K_EVAL,
                )
            )

            # -------------------------------------------------
            # Build V4 ranker feature matrix
            #
            # 64 user embedding
            # 64 item embedding
            #  6 user features
            #  6 item features
            #  1 MF score
            #  1 retrieval rank
            #
            # Total = 142
            # -------------------------------------------------
            feature_rows = []

            for retrieval_rank, item_id in enumerate(
                candidates,
                start=1,
            ):

                item_idx = item_to_idx[item_id]

                # ---------------------------------------------
                # Item embedding
                # ---------------------------------------------
                item_tensor = torch.tensor(
                    [item_idx],
                    dtype=torch.long,
                    device=device,
                )

                item_embedding = (
                    mf_model.item_embedding(item_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                # ---------------------------------------------
                # Item statistical features
                # ---------------------------------------------
                item_feat = item_feature_dict.get(
                    item_id,
                    np.zeros(
                        6,
                        dtype=np.float32,
                    ),
                )

                # ---------------------------------------------
                # MF interaction score
                # ---------------------------------------------
                mf_score = np.float32(
                    np.dot(
                        user_embedding,
                        item_embedding,
                    )
                )

                # ---------------------------------------------
                # Retrieval rank
                # ---------------------------------------------
                retrieval_rank_feature = np.float32(
                    retrieval_rank
                )

                feature_vector = np.concatenate(
                    [
                        user_embedding,
                        item_embedding,
                        user_feat,
                        item_feat,
                        np.asarray(
                            [mf_score],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            [retrieval_rank_feature],
                            dtype=np.float32,
                        ),
                    ]
                )

                # 64 + 64 + 6 + 6 + 1 + 1 = 142
                assert feature_vector.shape == (142,)

                feature_rows.append(
                    feature_vector
                )

            X = torch.tensor(
                np.asarray(
                    feature_rows,
                    dtype=np.float32,
                ),
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

            ranking_order = np.argsort(
                -ranker_scores
            )

            ranked_items = [
                candidates[i]
                for i in ranking_order
            ]

            ranker_top10 = ranked_items[:K_EVAL]

            ranker_recalls.append(
                recall_at_k(
                    ranker_top10,
                    [target_item],
                    K_EVAL,
                )
            )

            ranker_ndcgs.append(
                ndcg_at_k(
                    ranker_top10,
                    [target_item],
                    K_EVAL,
                )
            )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    total_users = len(validation_targets)

    candidate_recall = (
        candidate_hits / total_users
        if total_users > 0
        else 0.0
    )

    mf_recall = np.mean(mf_recalls)
    mf_ndcg = np.mean(mf_ndcgs)

    ranker_recall = np.mean(ranker_recalls)
    ranker_ndcg = np.mean(ranker_ndcgs)

    print()
    print("========================================")
    print("       V4 RANKER VALIDATION RESULTS")
    print("========================================")
    print()

    print(
        f"Validation users:      {total_users}"
    )

    print(
        f"Users evaluated:       {users_evaluated}"
    )

    print(
        f"Candidate Recall@500:  {candidate_recall:.4f}"
    )

    print()

    print("----------------------------------------")
    print("MF / FAISS")
    print("----------------------------------------")

    print(
        f"Recall@10:             {mf_recall:.4f}"
    )

    print(
        f"NDCG@10:               {mf_ndcg:.4f}"
    )

    print()

    print("----------------------------------------")
    print("Neural Ranker V4")
    print("----------------------------------------")

    print(
        f"Recall@10:             {ranker_recall:.4f}"
    )

    print(
        f"NDCG@10:               {ranker_ndcg:.4f}"
    )

    print()

    print("----------------------------------------")
    print("Improvement")
    print("----------------------------------------")

    if mf_recall > 0:
        recall_improvement = (
            (ranker_recall - mf_recall)
            / mf_recall
            * 100
        )
    else:
        recall_improvement = 0.0

    if mf_ndcg > 0:
        ndcg_improvement = (
            (ranker_ndcg - mf_ndcg)
            / mf_ndcg
            * 100
        )
    else:
        ndcg_improvement = 0.0

    print(
        f"Recall@10 improvement: "
        f"{recall_improvement:+.2f}%"
    )

    print(
        f"NDCG@10 improvement:   "
        f"{ndcg_improvement:+.2f}%"
    )

    print()


if __name__ == "__main__":
    main()
