from sqlalchemy import Column, Integer, Float, String, DateTime
from app.db import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)


class Item(Base):
    __tablename__ = "items"

    item_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)


class Interaction(Base):
    __tablename__ = "interactions"

    user_id = Column(Integer, primary_key=True)
    item_id = Column(Integer, primary_key=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
