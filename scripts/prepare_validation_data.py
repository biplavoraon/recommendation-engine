import pickle


INPUT_PATH = "data/ranker_data.pkl"
OUTPUT_PATH = "data/validation_data.pkl"


def main():
    with open(INPUT_PATH, "rb") as f:
        data = pickle.load(f)

    # All interactions available BEFORE validation.
    validation_history = data["ranker_train"].copy()

    validation_targets = data["ranker_validation"].copy()

    # User -> set of items available in validation history.
    validation_history_items = (
        validation_history
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    validation_targets = (
        validation_targets
        .set_index("user_id")["item_id"]
        .to_dict()
    )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "validation_history": validation_history,
                "validation_targets": validation_targets,
                "validation_history_items":
                    validation_history_items,
            },
            f,
        )

    print("=== Validation Data ===")
    print(
        f"Validation history: "
        f"{len(validation_history)}"
    )
    print(
        f"Validation targets: "
        f"{len(validation_targets)}"
    )
    print(
        f"Validation users: "
        f"{len(validation_history_items)}"
    )

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
