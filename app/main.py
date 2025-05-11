from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging
import os
from pathlib import Path
from .database import init_db
from .routes.auth_routes import router as auth_router
from .services.gemini_service import GeminiService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state"""
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Test Gemini API
        if not GeminiService().test_model():
            logger.error("Gemini API test failed")
            raise RuntimeError("Gemini API initialization failed")
        logger.info("Gemini API verified")
        
        # Load ML models
        model_dir = Path("app/model")
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_dir}")
            
        global models, vectorizers
        models = {
            "global": joblib.load(model_dir/"fake_news_model.pkl"),
            "india_data": joblib.load(model_dir/"india_data_model.pkl"),
            "india_titles": joblib.load(model_dir/"india_titles_model.pkl"),
        }
        vectorizers = {
            "global": joblib.load(model_dir/"tfidf_vectorizer.pkl"),
            "india_data": joblib.load(model_dir/"india_data_vectorizer.pkl"),
            "india_titles": joblib.load(model_dir/"india_titles_vectorizer.pkl"),
        }
        logger.info("ML models loaded successfully")
        
        # Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server initialization failed")
    
    yield
    logger.info("Application startup completed !!!")

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using ML models and Gemini verification",
    lifespan=lifespan
)

# Configure CORS (more secure in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
try:
    templates = Jinja2Templates(directory="app/templates")
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception as e:
    logger.error(f"Template/static files setup failed: {str(e)}")
    raise RuntimeError("Frontend resources configuration failed")

# Global variables for models and NLP
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class PredictRequest(BaseModel):
    title: str
    text: str
    region: str = "global"

def clean_text(text: str) -> str:
    """Preprocess text for ML model input"""
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    tokens = nltk.word_tokenize(text)
    return ' '.join(
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    )

@app.post("/predict/")
async def predict(
    article: PredictRequest, 
    background_tasks: BackgroundTasks
):
    """
    Analyze news article for authenticity
    Returns: {prediction: "REAL/FAKE", confidence: float, source: str}
    """
    if not article.title or not article.text:
        raise HTTPException(400, "Both title and text are required")
    
    try:
        # Preprocess text
        cleaned_text = clean_text(f"{article.title} {article.text}")
        if not cleaned_text:
            raise HTTPException(400, "Invalid text after cleaning")

        # Select model based on region
        region = article.region.lower()
        model_key = ("india_data" if region == "india" 
                    and article.text.strip() else "global")
        
        # Make prediction
        vectorized = vectorizers[model_key].transform([cleaned_text])
        prediction = models[model_key].predict(vectorized)[0]
        confidence = round(max(models[model_key].predict_proba(vectorized)[0]), 4)

        # Background verification with Gemini
        background_tasks.add_task(
            GeminiService().verify_content,
            article.title,
            article.text
        )

        return {
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": confidence,
            "source": "model"
        }
        
    except KeyError:
        raise HTTPException(400, "Invalid region specified")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Prediction service unavailable")

class GeminiVerifyRequest(BaseModel):
    title: str
    text: str

class GeminiVerifyRequest(BaseModel):
    title: str
    text: str

@app.post("/gemini-verify/")
async def gemini_verify(request: GeminiVerifyRequest):
    """Get Gemini's fact-checking analysis"""
    try:
        result = await GeminiService().verify_content(request.title, request.text)
        if not result:
            raise HTTPException(503, "Verification service unavailable")
            
        return {
            "inaccurate_phrases": result.get("inaccurate_phrases", []),
            "related_articles": result.get("related_articles", []),
            "verdict": result.get("verdict", "unverified")
        }
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise HTTPException(500, "Content verification failed")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve frontend entry point"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Endpoint for health monitoring"""
    return {"status": "healthy", "models_loaded": bool(models)}