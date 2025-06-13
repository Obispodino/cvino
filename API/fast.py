from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# 🔐 API Key for basic security
# API_KEY = "super-secret-key"  # ⚠️ Change this for production

app = FastAPI()

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔄 Load the trained model from pkl
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_model_Dino.pkl"))
    model = joblib.load(model_path)
    print("✅ Model loaded from trained_model_Dino.pkl")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# 🔹 Root endpoint
@app.get("/")
def root():
    return {"message": "Hi, the API is running!"}
