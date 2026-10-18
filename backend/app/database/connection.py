import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from backend.app.config import Config

class Base(DeclarativeBase):
    pass

# Ensure directory for SQLite exists if applicable
if Config.DATABASE_URL.startswith("sqlite"):
    db_file_dir = Config.DB_PATH.parent
    os.makedirs(db_file_dir, exist_ok=True)

engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False} if Config.DATABASE_URL.startswith("sqlite") else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from backend.app.database import models  # noqa
    Base.metadata.create_all(bind=engine)
