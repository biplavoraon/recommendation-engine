import numpy as np


def build_features(interactions):
    """
    Build user-level and item-level statistics using
    training interactions only.
    """

    interactions = interactions.copy()

    interactions["timestamp"] = (
        interactions["timestamp"]
        .astype("int64")
    )

    # --------------------------------------------------
    # Reference time
    # --------------------------------------------------

    reference_time = interactions["timestamp"].max()

    # --------------------------------------------------
    # User features
    # --------------------------------------------------

    user_stats = (
        interactions
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

    user_stats["user_days_since_last_interaction"] = (
        reference_time
        - user_stats["user_last_interaction"]
    ) / 86400.0

    user_stats["user_active_days"] = (
        user_stats["user_last_interaction"]
        - user_stats["user_first_interaction"]
    ) / 86400.0

    # --------------------------------------------------
    # Item features
    # --------------------------------------------------

    item_stats = (
        interactions
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
        reference_time
        - item_stats["item_last_interaction"]
    ) / 86400.0

    return user_stats, item_stats
