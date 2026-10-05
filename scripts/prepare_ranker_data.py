import pickle

import pandas as pd
from sqlalchemy import text

from app.db import engine


OUTPUT_PATH = "data/ranker_data.pkl"


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

    interactions = pd.read_sql(query, engine)

    # ---------------------------------------------------------
    # 1. Final evaluation split
    #    Latest interaction of every user = final test set
    # ---------------------------------------------------------
    final_test = (
        interactions
        .groupby("user_id", group_keys=False)
        .tail(1)
    )

    development = interactions.drop(final_test.index).copy()

    # ---------------------------------------------------------
    # 2. Inner temporal split for ranking
    #    Latest interaction in development = ranker validation
    # ---------------------------------------------------------
    ranker_validation = (
        development
        .groupby("user_id", group_keys=False)
        .tail(1)
    )

    ranker_train = development.drop(
        ranker_validation.index
    ).copy()

    # ---------------------------------------------------------
    # 3. Build user histories
    # ---------------------------------------------------------
    ranker_train_history = (
        ranker_train
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    ranker_validation_history = (
        development
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # ---------------------------------------------------------
    # 4. Save
    # ---------------------------------------------------------
    data = {
        "ranker_train": ranker_train,
        "ranker_validation": ranker_validation,
        "final_test": final_test,
        "ranker_train_history": ranker_train_history,
        "ranker_validation_history": ranker_validation_history,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)

    # ---------------------------------------------------------
    # 5. Print summary
    # ---------------------------------------------------------
    print("=== Ranker Data Preparation ===")

    print(f"Total interactions:       {len(interactions)}")
    print(f"Final test interactions:  {len(final_test)}")
    print(f"Development interactions: {len(development)}")
    print(f"Ranker train interactions:{len(ranker_train)}")
    print(f"Ranker validation:        {len(ranker_validation)}")

    print(
        f"Ranker train users:       "
        f"{ranker_train['user_id'].nunique()}"
    )

    print(
        f"Ranker validation users:  "
        f"{ranker_validation['user_id'].nunique()}"
    )

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
