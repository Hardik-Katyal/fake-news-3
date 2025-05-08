from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .auth import register_user, authenticate_user
from contextlib import asynccontextmanager
from .database import init_db
from .routes.auth_routes import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # Initialize DB on startup
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(auth_router, prefix="/api")  # Added /api prefix

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template and static files setup
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve HTML template at the root
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Load models and vectorizers for different regions
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

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

@app.post("/predict/")
async def predict(article: PredictRequest):
    # Log received article data for debugging
    print(f"Received Article: {article.title} {article.text}")

    # Validate that both title and text are provided
    if not article.title and not article.text:
        raise HTTPException(status_code=400, detail="Both title and text are required.")
    
    combined_text = article.title + " " + article.text
    cleaned_text = clean_text(combined_text)

    # Further validation to avoid empty or very short text input after cleaning
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Cleaned text is empty or invalid.")

    # Log cleaned text for debugging
    print(f"Cleaned Text: {cleaned_text}")

    region = article.region.lower()

    # Model selection based on region and input content
    if region == "india":
        if article.text.strip():  # If text (body content) is available, use india_data model
            model = models["india_data"]
            vectorizer = vectorizers["india_data"]
        else:  # If no body content, use india_titles model (just the title)
            model = models["india_titles"]
            vectorizer = vectorizers["india_titles"]
    else:
        # For all other regions (including US, global, etc.), use global model
        model = models.get(region, models["global"])  
        vectorizer = vectorizers.get(region, vectorizers["global"])

    # Vectorize the cleaned text and predict
    vectorized = vectorizer.transform([cleaned_text])
    print(f"Vectorized Text: {vectorized}")

    # Make the prediction
    prediction = model.predict(vectorized)[0]
    confidence = round(max(model.predict_proba(vectorized)[0]), 4)

    # Log prediction and confidence for debugging
    print(f"Prediction: {'REAL' if prediction == 1 else 'FAKE'}, Confidence: {confidence}")

    return {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": confidence
    }
