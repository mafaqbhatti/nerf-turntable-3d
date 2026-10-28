from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

# Use SQLite for local development (no PostgreSQL required)
# To use PostgreSQL, uncomment the line below and ensure PostgreSQL is running
# db_url = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/NeRF")
db_url = "sqlite:///./nerf_app.db"

engine = create_engine(db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
