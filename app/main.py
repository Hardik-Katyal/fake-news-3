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
from .database import init_db
from .routes.auth_routes import router as auth_router
from .services.gemini_service import GeminiService

# Initialize logging and environment
logger = logging.getLogger(__name__)
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state"""
    init_db()
    try:
        if not GeminiService().test_model():
            logger.error("Gemini API test failed")
            raise RuntimeError("Gemini API initialization failed")
    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}", exc_info=True)
        raise

app = FastAPI(lifespan=lifespan)
app.include_router(auth_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load ML models
try:
    models = {
        "global": joblib.load("app/model/fake_news_model.pkl"),
        "india_data": joblib.load("app/model/india_data_model.pkl"),
        "india_titles": joblib.load("app/model/india_titles_model.pkl"),
    }
    vectorizers = {
        "global": joblib.load("app/model/tfidf_vectorizer.pkl"),
        "india_data": joblib.load("app/model/india_data_vectorizer.pkl"),
        "india_titles": joblib.load("app/model/india_titles_vectorizer.pkl"),
    }
except Exception as e:
    logger.critical(f"Model loading failed: {str(e)}")
    raise RuntimeError("Failed to load ML models")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class PredictRequest(BaseModel):
    title: str
    text: str
    region: str

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    )

@app.post("/predict/")
async def predict(article: PredictRequest, background_tasks: BackgroundTasks):
    if not article.title or not article.text:
        raise HTTPException(400, "Both title and text are required")
    
    try:
        cleaned_text = clean_text(f"{article.title} {article.text}")
        if not cleaned_text:
            raise HTTPException(400, "Invalid text after cleaning")

        region = article.region.lower()
        model_key = "india_data" if region == "india" and article.text.strip() else "global"
        
        vectorized = vectorizers[model_key].transform([cleaned_text])
        prediction = models[model_key].predict(vectorized)[0]
        confidence = round(max(models[model_key].predict_proba(vectorized)[0]), 4)

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal prediction error")

@app.post("/gemini-verify/")
async def gemini_verify(title: str, text: str):
    try:
        result = await GeminiService().verify_content(title, text)
        return {
            "inaccurate_phrases": result.get("inaccurate_phrases", []),
            "related_articles": result.get("related_articles", [])
        }
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise HTTPException(500, "Content verification failed")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})