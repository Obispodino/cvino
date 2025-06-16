from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import joblib

from cv_functions.recommendation import get_wine_recommendations_by_characteristics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load precomputed metadata and model
try:
    metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw_data", "wine_metadata.csv"))
    wine_metadata_df = pd.read_csv(metadata_path)
    print("✅ Metadata loaded.")
except Exception as e:
    wine_metadata_df = None
    print(f"❌ Failed to load metadata: {e}")

class WineRequest(BaseModel):
    wine_type: str = "Red"
    grape_varieties: Optional[List[str]] = None
    body: str = "Full-bodied"
    abv: float = 12.0
    acidity: Optional[str] = None
    country: Optional[str] = None
    region_name: Optional[str] = None
    n_recommendations: int = 5

@app.get("/")
def root():
    return {"message": "Wine Recommender API is running."}

@app.post("/recommend-wines")
def recommend_wines(request: WineRequest):
    if wine_metadata_df is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")

    try:
        result_df = get_wine_recommendations_by_characteristics(
            wine_type=request.wine_type,
            grape_varieties=request.grape_varieties,
            body=request.body,
            abv=request.abv,
            acidity=request.acidity,
            country=request.country,
            region_name=request.region_name,
            n_recommendations=request.n_recommendations,
            metadata_df=wine_metadata_df
        )
        if result_df.empty:
            return {"message": "No recommendations found.", "wines": []}

        return {"wines": result_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
