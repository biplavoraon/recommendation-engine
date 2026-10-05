from pathlib import Path
from datetime import datetime

import pandas as pd

from app.db import SessionLocal
from app.models import User, Item, Interaction


DATA_DIR = Path("data/ml-100k")


def load_users(session):
    users = pd.read_csv(
        DATA_DIR / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    objects = [
        User(user_id=int(row.user_id))
        for row in users.itertuples()
    ]

    session.bulk_save_objects(objects)


def load_items(session):
    items = pd.read_csv(
        DATA_DIR / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"],
    )

    objects = [
        Item(
            item_id=int(row.item_id),
            title=row.title,
        )
        for row in items.itertuples()
    ]

    session.bulk_save_objects(objects)


def load_interactions(session):
    ratings = pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    objects = [
        Interaction(
            user_id=int(row.user_id),
            item_id=int(row.item_id),
            rating=float(row.rating),
            timestamp=datetime.fromtimestamp(int(row.timestamp)),
        )
        for row in ratings.itertuples()
    ]

    session.bulk_save_objects(objects)


def main():
    session = SessionLocal()

    try:
        print("Loading users...")
        load_users(session)

        print("Loading items...")
        load_items(session)

        print("Loading interactions...")
        load_interactions(session)

        session.commit()

        print("MovieLens successfully loaded.")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()
