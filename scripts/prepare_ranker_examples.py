import pickle

import pandas as pd


INPUT_PATH = "data/ranker_data.pkl"
OUTPUT_PATH = "data/ranker_examples.pkl"


def temporal_split(df):
    """
    For each user:
      - latest interaction = positive target
      - earlier interactions = candidate-generation history
    """
    positive = (
        df
        .groupby("user_id", group_keys=False)
        .tail(1)
    )

    history = df.drop(positive.index).copy()

    history_items = (
        history
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    positives = (
        positive
        .set_index("user_id")["item_id"]
        .to_dict()
    )

    return history, positives, history_items


def main():
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)

    ranker_train = data["ranker_train"]
    ranker_validation = data["ranker_validation"]
    final_test = data["final_test"]

    # ---------------------------------------------------------
    # Training examples
    # ---------------------------------------------------------
    (
        train_history,
        train_positive,
        train_history_items,
    ) = temporal_split(ranker_train)

    # ---------------------------------------------------------
    # Validation examples
    #
    # We already have a temporal validation interaction.
    # Its history is the complete development set.
    # ---------------------------------------------------------
    validation_positive = (
        ranker_validation
        .set_index("user_id")["item_id"]
        .to_dict()
    )

    validation_history_items = data[
        "ranker_validation_history"
    ]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    examples = {
        "train_history": train_history,
        "train_positive": train_positive,
        "train_history_items": train_history_items,

        "validation_positive": validation_positive,
        "validation_history_items": validation_history_items,

        "final_test": final_test,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(examples, f)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("=== Ranker Example Preparation ===")

    print(
        f"Ranker training interactions: "
        f"{len(ranker_train)}"
    )

    print(
        f"Ranker training history:       "
        f"{len(train_history)}"
    )

    print(
        f"Ranker training positives:      "
        f"{len(train_positive)}"
    )

    print(
        f"Training users:                 "
        f"{len(train_positive)}"
    )

    print(
        f"Validation users:               "
        f"{len(validation_positive)}"
    )

    print(
        f"Final test interactions:        "
        f"{len(final_test)}"
    )

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
