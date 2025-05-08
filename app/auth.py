from passlib.context import CryptContext
from sqlalchemy.orm import Session
from .models import User
from .database import get_db
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise ValueError("Password processing failed")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_user(db: Session, username: str, email: str, password: str):
    try:
        hashed = get_password_hash(password)
        user = User(username=username, email=email, password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"User Registered: {user.username}, Email: {user.email}, ID: {user.id}")
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"User creation failed: {str(e)}")
        raise

def authenticate_user(db: Session, username: str, password: str):
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.password):
            return None
        logger.info(f"User Logged In: {user.username}, ID: {user.id}")
        return user
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

def register_user(db: Session, username: str, email: str, password: str):
    try:
        existing_user = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            if existing_user.username == username:
                raise ValueError("Username already exists")
            raise ValueError("Email already exists")
            
        return create_user(db, username, email, password)
    except Exception as e:
        db.rollback()
        logger.error(f"Registration failed: {str(e)}")
        raise