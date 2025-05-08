from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)  # Added email field
    password = Column(String)

    entries = relationship("PredictionEntry", back_populates="owner")

class PredictionEntry(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_text = Column(Text)
    result = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="entries")