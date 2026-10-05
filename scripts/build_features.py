import pickle

import pandas as pd
from sqlalchemy import text

from app.db import engine
from app.ml.features import build_features


OUTPUT_PATH = "data/features.pkl"


def main():
    query = text(
        """
        SELECT
            user_id,
            item_id,
            rating,
            timestamp
        FROM interactions
        ORDER BY user_id, timestamp
        """
    )

    interactions = pd.read_sql(
        query,
        engine,
    )

    print(
        f"Total interactions: "
        f"{len(interactions)}"
    )

    # --------------------------------------------------
    # Temporal split
    # --------------------------------------------------

    test = (
        interactions
        .groupby("user_id", group_keys=False)
        .tail(1)
    )

    train = interactions.drop(
        test.index
    ).copy()

    print(
        f"Training interactions: "
        f"{len(train)}"
    )

    print(
        f"Test interactions: "
        f"{len(test)}"
    )

    # --------------------------------------------------
    # Build features ONLY from training data
    # --------------------------------------------------

    user_stats, item_stats = build_features(
        train
    )

    print(
        f"Users with features: "
        f"{len(user_stats)}"
    )

    print(
        f"Items with features: "
        f"{len(item_stats)}"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "user_stats": user_stats,
                "item_stats": item_stats,
                "train_interactions": train,
                "test_interactions": test,
            },
            f,
        )

    print(
        f"Saved training-only features "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
