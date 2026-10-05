from sqlalchemy import func

from app.models import Interaction


class PopularityRecommender:

    def __init__(self, session):
        self.session = session

        self.popular_items = self._compute_popularity()

    def _compute_popularity(self):
        results = (
            self.session.query(
                Interaction.item_id,
                func.count(Interaction.user_id).label("interaction_count"),
            )
            .group_by(Interaction.item_id)
            .order_by(
                func.count(Interaction.user_id).desc()
            )
            .all()
        )

        return [item_id for item_id, _ in results]

    def recommend(self, user_id, k=10):
        interacted_items = {
            item_id
            for (item_id,) in (
                self.session.query(Interaction.item_id)
                .filter(Interaction.user_id == user_id)
                .all()
            )
        }

        recommendations = []

        for item_id in self.popular_items:

            if item_id not in interacted_items:
                recommendations.append(item_id)

            if len(recommendations) == k:
                break

        return recommendations
