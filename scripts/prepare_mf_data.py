import pickle
from collections import defaultdict

from sqlalchemy import desc

from app.db import SessionLocal
from app.models import Interaction, Item


def build_train_data(session):

    interactions = (
        session.query(Interaction)
        .order_by(
            Interaction.user_id,
            desc(Interaction.timestamp),
        )
        .all()
    )

    user_interactions = defaultdict(list)

    for interaction in interactions:
        user_interactions[interaction.user_id].append(
            interaction.item_id
        )

    train_items = {}
    test_items = {}

    for user_id, items in user_interactions.items():

        # Most recent interaction is test.
        test_items[user_id] = items[0]

        # Older interactions are training.
        train_items[user_id] = items[1:]

    return train_items, test_items


def create_mappings(train_items, all_item_ids):

    user_ids = sorted(train_items.keys())

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    item_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(all_item_ids)
    }

    return user_to_idx, item_to_idx
    

def convert_to_indices(
    train_items,
    user_to_idx,
    item_to_idx,
):

    converted = {}

    for user_id, items in train_items.items():

        user_idx = user_to_idx[user_id]

        converted[user_idx] = {
            item_to_idx[item_id]
            for item_id in items
            if item_id in item_to_idx
        }

    return converted


def main():

    session = SessionLocal()

    try:

        train_items, test_items = build_train_data(
            session
        )

        all_items = (
            session.query(Item.item_id)
            .order_by(Item.item_id)
            .all()
        )

        all_item_ids = [
            item_id
            for (item_id,) in all_items
        ]

        user_to_idx, item_to_idx = create_mappings(
            train_items,
            all_item_ids,
        )

        train_indices = convert_to_indices(
            train_items,
            user_to_idx,
            item_to_idx,
        )

        data = {
            "train_items": train_indices,
            "test_items": test_items,
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
        }

        with open(
            "data/mf_data.pkl",
            "wb",
        ) as f:
            pickle.dump(data, f)

        print(
            f"Users: {len(user_to_idx)}"
        )

        print(
            f"Items: {len(item_to_idx)}"
        )

        print(
            f"Training users: {len(train_indices)}"
        )

        print(
            "Saved data/mf_data.pkl"
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()
