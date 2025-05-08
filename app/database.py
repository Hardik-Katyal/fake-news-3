from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///./fakenews.db"

# Initialize database engine with connection pooling and timeout
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30  # 30 second timeout
    },
    pool_pre_ping=True  # Check connections before using them
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

def init_db():
    try:
        logger.info("Initializing database tables")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def get_db():
    db = SessionLocal()
    try:
        logger.debug("Database session started")
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        logger.debug("Database session closed")
        db.close()