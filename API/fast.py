from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import joblib

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO



from cv_functions.recommendation import get_wine_recommendations_by_characteristics
from cv_functions.food_recommendation import get_wine_recommendations_by_food
from cv_functions.wine_label_ai2 import extract_wine_info_from_image
from cv_functions.model import load_model

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
    # Load model path
    LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_model.pkl"))
    model = load_model(LOCAL_MODEL_PATH)
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


class FoodWineRequest(BaseModel):
    food_pairing: str
    wine_type: Optional[str] = None
    grape_varieties: Optional[List[str]] = None
    body: Optional[str] = None
    abv: Optional[float] = None
    acidity: Optional[str] = None
    country: Optional[str] = None
    region_name: Optional[str] = None
    n_recommendations: int = 5
    exact_match_only: bool = False


@app.get("/")
def root():
    return {"message": "Wine Recommender API is running."}

@app.post("/recommend-wines")
def recommend_wines(request: WineRequest):
    if wine_metadata_df is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")

# try:
    result_df = get_wine_recommendations_by_characteristics(
        wine_type=request.wine_type,
        grape_varieties=request.grape_varieties,
        body=request.body,
        abv=request.abv,
        acidity=request.acidity,
        country=request.country,
        region_name=request.region_name,
        n_recommendations=request.n_recommendations,
        metadata_df=wine_metadata_df,
        model=model
    )
    if result_df.empty:
        return {"message": "No recommendations found.", "wines": []}

    return {"wines": result_df.to_dict(orient="records")}
# except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/recommend-by-food")
def recommend_by_food(request: FoodWineRequest):
    if wine_metadata_df is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")

    try:
        result_df = get_wine_recommendations_by_food(
            features_df=wine_metadata_df,
            food_pairing=request.food_pairing,
            wine_type=request.wine_type,
            grape_varieties=request.grape_varieties,
            body=request.body,
            abv=request.abv,
            acidity=request.acidity,
            country=request.country,
            region_name=request.region_name,
            n_recommendations=request.n_recommendations,
            exact_match_only=request.exact_match_only
        )
        if result_df.empty:
            return {"message": f"No wines found for food '{request.food_pairing}'.", "wines": []}

        return {"wines": result_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Food recommendation failed: {e}")


@app.post('/read_image')
async def receive_image(img: UploadFile = File(...)):
    try:
        # Step 1: Read bytes from uploaded image
        contents = await img.read()  # type: bytes

        # Step 2: Convert bytes to image (PIL)
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Step 3: Preprocess image (e.g., resize)
        wine_info = extract_wine_info_from_image(image)

        return wine_info
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
