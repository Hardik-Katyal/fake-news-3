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
from .database import init_db
from .routes.auth_routes import router as auth_router
from .services.gemini_service import GeminiService

# Load environment variables first
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state"""
    init_db()
    # Test Gemini connection on startup
    try:
        gemini_service = GeminiService()
        if not gemini_service.test_model():
            raise RuntimeError("Gemini API test failed")
    except Exception as e:
        raise RuntimeError(f"Could not initialize Gemini: {str(e)}")
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(auth_router, prefix="/api")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
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
    raise RuntimeError(f"Failed to load ML models: {str(e)}")

# Initialize NLP resources
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
    """Preprocess text for model input"""
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic
    text = re.sub(r'\s+', ' ', text)      # Collapse whitespace
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
    """Main prediction endpoint"""
    if not article.title or not article.text:
        raise HTTPException(
            status_code=400, 
            detail="Both title and text are required"
        )
    
    try:
        # Text processing
        cleaned_text = clean_text(f"{article.title} {article.text}")
        if not cleaned_text:
            raise HTTPException(
                status_code=400, 
                detail="Invalid text after cleaning"
            )

        # Model selection
        region = article.region.lower()
        model_key = "india_data" if region == "india" and article.text.strip() else "global"
        
        # Prediction
        vectorized = vectorizers[model_key].transform([cleaned_text])
        prediction = models[model_key].predict(vectorized)[0]
        confidence = round(max(models[model_key].predict_proba(vectorized)[0]), 4)

        # Async verification
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
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction service error")

@app.post("/gemini-verify/")
async def gemini_verify(title: str, text: str):
    """Direct verification endpoint"""
    try:
        result = await GeminiService().verify_content(title, text)
        return {
            "inaccurate_phrases": result.get("inaccurate_phrases", []),
            "related_articles": result.get("related_articles", [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve frontend"""
    return templates.TemplateResponse("index.html", {"request": request})