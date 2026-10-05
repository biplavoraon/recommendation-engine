import pickle

import numpy as np
import pandas as pd

INPUT_PATH = "data/ranker_examples.pkl"
OUTPUT_PATH = "data/ranker_features.pkl"


def main():
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)

    train_history = data["train_history"].copy()

    # Make sure timestamp is a datetime column.
    train_history["timestamp"] = pd.to_datetime(
        train_history["timestamp"]
    )

    # ---------------------------------------------------------
    # User statistics
    # ---------------------------------------------------------
    user_stats = (
        train_history
        .groupby("user_id")
        .agg(
            user_interaction_count=("item_id", "count"),
            user_avg_rating=("rating", "mean"),
            user_rating_std=("rating", "std"),
            user_unique_items=("item_id", "nunique"),
            user_last_interaction=("timestamp", "max"),
            user_first_interaction=("timestamp", "min"),
        )
        .reset_index()
    )

    user_stats["user_rating_std"] = (
        user_stats["user_rating_std"]
        .fillna(0.0)
    )

    reference_time = train_history["timestamp"].max()

    user_stats["user_days_since_last_interaction"] = (
        (
            reference_time
            - user_stats["user_last_interaction"]
        )
        .dt.total_seconds()
        / 86400.0
    )

    user_stats["user_active_days"] = (
        (
            user_stats["user_last_interaction"]
            - user_stats["user_first_interaction"]
        )
        .dt.total_seconds()
        / 86400.0
    )

    # ---------------------------------------------------------
    # Item statistics
    # ---------------------------------------------------------
    item_stats = (
        train_history
        .groupby("item_id")
        .agg(
            item_interaction_count=("user_id", "count"),
            item_avg_rating=("rating", "mean"),
            item_rating_std=("rating", "std"),
            item_unique_users=("user_id", "nunique"),
            item_last_interaction=("timestamp", "max"),
        )
        .reset_index()
    )

    item_stats["item_rating_std"] = (
        item_stats["item_rating_std"]
        .fillna(0.0)
    )

    item_stats["item_log_popularity"] = np.log1p(
        item_stats["item_interaction_count"]
    )

    item_stats["item_days_since_last_interaction"] = (
        (
            reference_time
            - item_stats["item_last_interaction"]
        )
        .dt.total_seconds()
        / 86400.0
    )

    # ---------------------------------------------------------
    # Remove raw timestamp columns.
    # ---------------------------------------------------------
    user_stats = user_stats.drop(
        columns=[
            "user_last_interaction",
            "user_first_interaction",
        ]
    )

    item_stats = item_stats.drop(
        columns=[
            "item_last_interaction",
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "user_stats": user_stats,
                "item_stats": item_stats,
            },
            f,
        )

    print("=== Ranker Feature Preparation ===")
    print(
        f"Training interactions: {len(train_history)}"
    )
    print(
        f"Users with features: {len(user_stats)}"
    )
    print(
        f"Items with features: {len(item_stats)}"
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
