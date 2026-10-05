from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Interaction
from app.recommender_mf import MFRecommender

import json
import redis
from app.config import settings

redis_client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
)


app = FastAPI(
    title="Recommendation Engine",
    version="1.0.0",
)


# Load model once when the application starts.
recommender = MFRecommender()


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    if k <= 0 or k > 100:
        raise HTTPException(
            status_code=400,
            detail="k must be between 1 and 100",
        )

    cache_key = f"recommendations:{user_id}:{k}"

    # Check Redis
    cached = redis_client.get(cache_key)

    if cached is not None:
        return {
            "user_id": user_id,
            "recommendations": json.loads(cached),
        }

    db: Session = SessionLocal()

    try:
        user_exists = (
            db.query(Interaction.user_id)
            .filter(Interaction.user_id == user_id)
            .first()
        )

        if user_exists is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        interacted_items = {
            item_id
            for (item_id,) in (
                db.query(Interaction.item_id)
                .filter(Interaction.user_id == user_id)
                .all()
            )
        }

        recommendations = recommender.recommend(
            user_id=user_id,
            interacted_items=interacted_items,
            k=k,
        )

        # Cache for 5 minutes
        redis_client.setex(
            cache_key,
            300,
            json.dumps(recommendations),
        )

        return {
            "user_id": user_id,
            "recommendations": recommendations,
        }

    finally:
        db.close()
