import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization


EXAMPLES_PATH = "data/ranker_examples.pkl"
FEATURES_PATH = "data/ranker_features.pkl"

MODEL_PATH = "models/ranker_mf_model.pt"
INDEX_PATH = "models/ranker_item_index.faiss"

OUTPUT_PATH = "data/ranker_dataset.pkl"

NUM_CANDIDATES = 500
NUM_NEGATIVES = 20


def main():
    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    with open(EXAMPLES_PATH, "rb") as f:
        examples = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    index = faiss.read_index(INDEX_PATH)

    model = MatrixFactorization(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    user_to_idx = checkpoint["user_to_idx"]
    item_to_idx = checkpoint["item_to_idx"]

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    # ---------------------------------------------------------
    # Feature lookup tables
    # ---------------------------------------------------------
    user_stats = (
        features["user_stats"]
        .set_index("user_id")
    )

    item_stats = (
        features["item_stats"]
        .set_index("item_id")
    )

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
    # Storage
    # ---------------------------------------------------------
    X = []
    y = []

    users_used = 0
    positives_found = 0

    # ---------------------------------------------------------
    # Build examples
    # ---------------------------------------------------------
    for user_id, positive_item_id in examples[
        "train_positive"
    ].items():

        if user_id not in user_to_idx:
            continue

        if positive_item_id not in item_to_idx:
            continue

        user_idx = user_to_idx[user_id]

        # -----------------------------------------------------
        # User embedding
        # -----------------------------------------------------
        user_tensor = torch.tensor(
            [user_idx],
            dtype=torch.long,
        )

        with torch.no_grad():
            user_embedding = (
                model.user_embedding(user_tensor)
                .cpu()
                .numpy()[0]
                .astype(np.float32)
            )

        # -----------------------------------------------------
        # FAISS retrieval
        # -----------------------------------------------------
        history_items = examples[
            "train_history_items"
        ][user_id]

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

            # Never use previously observed items.
            if item_id in history_items:
                continue

            candidates.append(
                (item_id, float(score))
            )

            if len(candidates) >= NUM_CANDIDATES:
                break

        # -----------------------------------------------------
        # Positive must be present
        # -----------------------------------------------------
        positive_position = None

        for position, (item_id, _) in enumerate(candidates):
            if item_id == positive_item_id:
                positive_position = position
                break

        if positive_position is None:
            continue

        positives_found += 1

        # -----------------------------------------------------
        # Select hard negatives
        #
        # Candidates nearest to the positive's retrieval
        # ranking are difficult examples for the ranker.
        # -----------------------------------------------------
        negative_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] != positive_item_id
        ]

        negative_candidates = negative_candidates[
            :NUM_NEGATIVES
        ]

        # -----------------------------------------------------
        # Common user feature vector
        # -----------------------------------------------------
        user_features = (
            user_stats.loc[
                user_id,
                user_feature_columns,
            ]
            .to_numpy(dtype=np.float32)
        )

        # -----------------------------------------------------
        # Positive + negatives
        # -----------------------------------------------------
        selected_candidates = [
            (positive_item_id, None)
        ] + negative_candidates

        for item_id, retrieval_score in selected_candidates:

            # Item statistics may not exist for catalog items
            # with no interaction in the ranker history.
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
                    len(item_feature_columns),
                    dtype=np.float32,
                )

            # -------------------------------------------------
            # Item embedding
            # -------------------------------------------------
            item_idx = item_to_idx[item_id]

            item_embedding = (
                model.item_embedding.weight[
                    item_idx
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # -------------------------------------------------
            # MF score
            # -------------------------------------------------
            mf_score = float(
                np.dot(
                    user_embedding,
                    item_embedding,
                )
            )

            # -------------------------------------------------
            # Feature vector
            #
            # 64 user embedding
            # 64 item embedding
            # 6 user features
            # 6 item features
            # 1 MF score
            #
            # Total = 141
            # -------------------------------------------------
            feature_vector = np.concatenate(
                [
                    user_embedding,
                    item_embedding,
                    user_features,
                    item_features,
                    np.array(
                        [mf_score],
                        dtype=np.float32,
                    ),
                ]
            )

            X.append(feature_vector)

            y.append(
                1.0
                if item_id == positive_item_id
                else 0.0
            )

        users_used += 1

    # ---------------------------------------------------------
    # Convert to arrays
    # ---------------------------------------------------------
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "X": X,
                "y": y,
                "feature_dim": X.shape[1],
                "users_used": users_used,
                "positives_found": positives_found,
            },
            f,
        )

    print("=== Ranker Dataset ===")
    print(f"Users used:       {users_used}")
    print(f"Positives found:  {positives_found}")
    print(f"Examples:         {len(X)}")
    print(f"Feature dimension:{X.shape[1]}")
    print(f"Positive labels:  {int(y.sum())}")
    print(
        f"Negative labels:  {int((y == 0).sum())}"
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
