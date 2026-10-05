from app.db import Base, engine
from app.models import User, Item, Interaction


Base.metadata.create_all(engine)

print("Database tables created.")
