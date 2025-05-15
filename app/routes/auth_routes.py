from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ..database import get_db
from ..auth import register_user, authenticate_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Authentication"])  # Changed prefix to /api

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

@router.post("/register/")
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        logger.info(f"Registration attempt for {user_data.username}")
        user = register_user(
            db, 
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        logger.info(f"Successfully registered {user.username}")
        return {
            "message": "Registration successful",
            "username": user.username,
            "email": user.email
        }
    except ValueError as e:
        logger.warning(f"Registration validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Registration failed. Please try again later."
        )

class UserLogin(BaseModel):
    username: str
    password: str

@router.post("/login/")
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    try:
        user = authenticate_user(db, user_data.username, user_data.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        return {
            "message": "Login successful",
            "user": {
                "username": user.username,
                "email": user.email
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
