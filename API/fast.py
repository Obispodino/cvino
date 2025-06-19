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
    app.state.wine_metadata_df = pd.read_csv(metadata_path)
    print("✅ Metadata loaded.")

    # Load model path
    LOCAL_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "trained_model.pkl"))
    app.state.model = load_model(LOCAL_MODEL_PATH)
    print("✅ Model loaded successfully!")

    if app.state.model is None:
        print("❌ Warning: Model loaded but is None!")
except Exception as e:
    app.state.wine_metadata_df = None
    app.state.model = None
    print(f"❌ Failed to load metadata or model: {e}")

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
    if app.state.wine_metadata_df is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")


    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert "None" or "string" strings to None/null values
        acidity = None if request.acidity in ["None", "string"] else request.acidity
        country = None if request.country in ["None", "string"] else request.country
        region_name = None if request.region_name in ["None", "string"] else request.region_name

        result_df = get_wine_recommendations_by_characteristics(
            wine_type=request.wine_type,
            grape_varieties=request.grape_varieties,
            body=request.body,
            abv=request.abv,
            acidity=acidity,
            country=country,
            region_name=region_name,
            n_recommendations=request.n_recommendations,
            metadata_df=app.state.wine_metadata_df,
            model=app.state.model
        )

        # Check if result_df is None first
        if result_df is None:
            return {"message": "No recommendations could be generated.", "wines": []}

        if result_df.empty:
            return {"message": "No recommendations found.", "wines": []}

        return {"wines": result_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/recommend-by-food")
def recommend_by_food(request: FoodWineRequest):
    if app.state.wine_metadata_df is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded")

    try:
        # Convert "string" values to None
        wine_type = None if request.wine_type == "string" else request.wine_type
        body = None if request.body == "string" else request.body
        acidity = None if request.acidity == "string" else request.acidity
        country = None if request.country == "string" else request.country
        region_name = None if request.region_name == "string" else request.region_name

        # If grape_varieties contains only "string", set to None
        grape_varieties = None
        if request.grape_varieties and len(request.grape_varieties) > 0:
            if len(request.grape_varieties) == 1 and request.grape_varieties[0] == "string":
                grape_varieties = None
            else:
                grape_varieties = request.grape_varieties

        result_df = get_wine_recommendations_by_food(
            features_df=app.state.wine_metadata_df,
            food_pairing=request.food_pairing,
            wine_type=wine_type,
            grape_varieties=grape_varieties,
            body=body,
            abv=request.abv,
            acidity=acidity,
            country=country,
            region_name=region_name,
            n_recommendations=request.n_recommendations,
            exact_match_only=request.exact_match_only
        )
        if result_df.empty:
            return {"message": f"No wines found that pair with '{request.food_pairing}'.", "wines": []}

        return {"wines": result_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Food recommendation failed: {e}")


# @app.post('/read_image')
# async def receive_image(img: UploadFile = File(...)):
#     try:
#         # Step 1: Read bytes from uploaded image
#         contents = await img.read()  # type: bytes

#         # Step 2: Convert bytes to image (PIL)
#         image = Image.open(BytesIO(contents)).convert("RGB")

#         # Step 3: Preprocess image (e.g., resize)
#         wine_info = extract_wine_info_from_image(image)

#         return wine_info
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@app.post('/read_image')
async def receive_image(img: UploadFile = File(...), n_recommendations: int = 5):
    try:
        # Step 1: Read bytes from uploaded image
        contents = await img.read()  # type: bytes

        # Step 2: Convert bytes to image (PIL)
        image = Image.open(BytesIO(contents)).convert("RGB")

        # Step 3: Extract wine info from image
        wine_info = extract_wine_info_from_image(image)

        # Step 4: Get recommendations based on extracted info
        if wine_info["extraction_successful"]:
            try:
                # Parse ABV to float
                abv = float(wine_info["ABV"]) if wine_info["ABV"] and wine_info["ABV"].replace('.', '', 1).isdigit() else 12.0

                # Get recommendations
                result_df = get_wine_recommendations_by_characteristics(
                    wine_type=wine_info["wine_type"],
                    grape_varieties=wine_info["grape_varieties"],
                    body=wine_info["body"],
                    abv=abv,
                    acidity=wine_info["acidity"],
                    country=wine_info["country"],
                    region_name=wine_info["region"],
                    n_recommendations=n_recommendations,
                    metadata_df=app.state.wine_metadata_df,
                    model=app.state.model
                )

                # Include recommendations in response
                if result_df is not None and not result_df.empty:
                    wine_info["recommendations"] = result_df.to_dict(orient="records")
                else:
                    wine_info["recommendations"] = []
                    wine_info["recommendation_message"] = "No similar wines found."
            except Exception as e:
                wine_info["recommendations"] = []
                wine_info["recommendation_error"] = str(e)
        else:
            wine_info["recommendations"] = []
            wine_info["recommendation_message"] = "Could not generate recommendations because wine info extraction failed."

        return wine_info
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})


@app.get("/check-model")
def check_model():
    return {
        "model_loaded": app.state.model is not None,
        "metadata_loaded": app.state.wine_metadata_df is not None,
        "metadata_shape": str(app.state.wine_metadata_df.shape) if app.state.wine_metadata_df is not None else None,
    }
