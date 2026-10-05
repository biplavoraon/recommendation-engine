from collections import Counter, defaultdict

from sqlalchemy import desc

from app.db import SessionLocal
from app.models import Interaction
from app.evaluation import recall_at_k, ndcg_at_k


K = 10


def build_train_test(session):
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
        user_interactions[interaction.user_id].append(interaction)

    train_items = {}
    test_items = {}

    for user_id, interactions in user_interactions.items():

        # Most recent interaction → test
        test_interaction = interactions[0]

        test_items[user_id] = test_interaction.item_id

        # Everything else → training
        train_items[user_id] = {
            x.item_id
            for x in interactions[1:]
        }

    return train_items, test_items


def build_popularity(train_items):
    popularity = Counter()

    for items in train_items.values():
        popularity.update(items)

    return [
        item_id
        for item_id, _ in popularity.most_common()
    ]


def recommend(popular_items, seen_items, k):
    recommendations = []

    for item_id in popular_items:

        if item_id not in seen_items:
            recommendations.append(item_id)

        if len(recommendations) == k:
            break

    return recommendations


def main():

    session = SessionLocal()

    try:
        train_items, test_items = build_train_test(session)

        popular_items = build_popularity(train_items)

        recalls = []
        ndcgs = []

        for user_id, test_item in test_items.items():

            recommendations = recommend(
                popular_items,
                train_items[user_id],
                K,
            )

            relevant = [test_item]

            recalls.append(
                recall_at_k(
                    recommendations,
                    relevant,
                    K,
                )
            )

            ndcgs.append(
                ndcg_at_k(
                    recommendations,
                    relevant,
                    K,
                )
            )

        print(f"Users evaluated: {len(test_items)}")
        print(f"Recall@{K}: {sum(recalls) / len(recalls):.4f}")
        print(f"NDCG@{K}:   {sum(ndcgs) / len(ndcgs):.4f}")

    finally:
        session.close()


if __name__ == "__main__":
    main()
