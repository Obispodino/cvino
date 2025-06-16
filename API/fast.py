from fastapi import FastAPI, Header, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import os
import pandas as pd
import tempfile
import shutil
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom functions
from cv_functions.wine_label_ai import extract_wine_info_from_image
from cv_functions.recommendation import get_wine_recommendations_by_characteristics

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

# Load wine dataset for recommendations
try:
    LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "Obispodino", "cvino", "raw_data")
    wine_data_path = os.path.join(LOCAL_DATA_PATH, 'wine_lookup.csv')
    wine_df = pd.read_csv(wine_data_path)
    print(f"✅ Wine data loaded from {wine_data_path}")
except Exception as e:
    wine_df = None
    print(f"⚠️ Could not load wine data: {e}")

# 🔄 Load the trained model from pkl
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_model_Dino.pkl"))
    model = joblib.load(model_path)
    print("✅ Model loaded from trained_model_Dino.pkl")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# Pydantic models for request/response validation
class WineCharacteristics(BaseModel):
    wine_type: str
    grape_varieties: Optional[List[str]] = None
    body: Optional[str] = None
    acidity: Optional[str] = None
    abv: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    n_recommendations: int = 5

class WineInfo(BaseModel):
    wine_type: Optional[str] = None
    grape_varieties: Optional[List[str]] = None
    body: Optional[str] = None
    acidity: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    ABV: Optional[float] = None
    extraction_successful: bool
    error: Optional[str] = None

class WineRecommendation(BaseModel):
    wine_id: str
    name: str
    country: Optional[str] = None
    wine_type: Optional[str] = None
    region: Optional[str] = None
    body: Optional[str] = None
    acidity: Optional[str] = None
    grapes: Optional[List[str]] = None
    rating: Optional[float] = None
    similarity: Optional[float] = None

class WineRecommendationsResponse(BaseModel):
    recommendations: List[WineRecommendation]
    extracted_info: Optional[WineInfo] = None

# 🔹 Root endpoint
@app.get("/")
def root():
    return {"message": "Hi, the Wine API is running!"}

# Endpoint for wine recommendations based on characteristics
@app.post("/recommendations", response_model=List[WineRecommendation])
async def get_recommendations(wine_characteristics: WineCharacteristics):
    """
    Get wine recommendations based on specified characteristics
    """
    if wine_df is None:
        raise HTTPException(status_code=500, detail="Wine data not available")

    try:
        # Call the recommendation function
        recommendations = get_wine_recommendations_by_characteristics(
            df=wine_df,
            wine_type=wine_characteristics.wine_type,
            grape_varieties=wine_characteristics.grape_varieties,
            body=wine_characteristics.body,
            abv=wine_characteristics.abv if wine_characteristics.abv else 12.0,
            acidity=wine_characteristics.acidity,
            country=wine_characteristics.country,
            region_name=wine_characteristics.region,
            n_recommendations=wine_characteristics.n_recommendations
        )

        if recommendations.empty:
            return []

        # Format the response
        result = []
        for _, wine in recommendations.iterrows():
            grapes_list = wine['Grapes_list']
            if isinstance(grapes_list, str):
                import ast
                try:
                    grapes_list = ast.literal_eval(grapes_list)
                except:
                    grapes_list = []

            result.append(WineRecommendation(
                wine_id=str(wine['WineID']),
                name=wine['WineName'],
                country=wine['Country'],
                wine_type=wine['Type'],
                region=wine['RegionName'],
                body=wine['Body'],
                acidity=wine['Acidity'],
                grapes=grapes_list,
                rating=float(wine['avg_rating']) if 'avg_rating' in wine else None,
                similarity=float(wine['Similarity']) if 'Similarity' in wine else None
            ))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Endpoint for wine label image analysis
@app.post("/analyze-label", response_model=WineInfo)
async def analyze_wine_label(file: UploadFile = File(...)):
    """
    Analyze a wine label image and extract information
    """
    # Create a temporary file to store the uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        # Process the image with the wine label AI
        result = extract_wine_info_from_image(temp_file.name)

        # Return the extracted information
        return WineInfo(
            wine_type=result.get('wine_type'),
            grape_varieties=result.get('grape_varieties'),
            body=result.get('body'),
            acidity=result.get('acidity'),
            country=result.get('country'),
            region=result.get('region'),
            ABV=result.get('ABV'),
            extraction_successful=result.get('extraction_successful', False),
            error=result.get('error')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Endpoint for wine recommendations from label image
@app.post("/recommendations-from-label", response_model=WineRecommendationsResponse)
async def get_recommendations_from_label(
    file: UploadFile = File(...),
    n_recommendations: int = Form(5)
):
    """
    Extract information from a wine label image and return wine recommendations
    """
    if wine_df is None:
        raise HTTPException(status_code=500, detail="Wine data not available")

    # Create a temporary file to store the uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        # Process the image with the wine label AI
        extracted_info = extract_wine_info_from_image(temp_file.name)

        if not extracted_info.get('extraction_successful', False):
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract information from the wine label: {extracted_info.get('error', 'Unknown error')}"
            )

        # Get wine recommendations based on the extracted information
        recommendations = get_wine_recommendations_by_characteristics(
            df=wine_df,
            wine_type=extracted_info.get('wine_type'),
            grape_varieties=extracted_info.get('grape_varieties'),
            body=extracted_info.get('body'),
            abv=extracted_info.get('ABV', 12.0),
            acidity=extracted_info.get('acidity'),
            country=extracted_info.get('country'),
            region_name=extracted_info.get('region'),
            n_recommendations=n_recommendations
        )

        # Format the response
        result = []
        if not recommendations.empty:
            for _, wine in recommendations.iterrows():
                grapes_list = wine['Grapes_list']
                if isinstance(grapes_list, str):
                    import ast
                    try:
                        grapes_list = ast.literal_eval(grapes_list)
                    except:
                        grapes_list = []

                result.append(WineRecommendation(
                    wine_id=str(wine['WineID']),
                    name=wine['WineName'],
                    country=wine['Country'],
                    wine_type=wine['Type'],
                    region=wine['RegionName'],
                    body=wine['Body'],
                    acidity=wine['Acidity'],
                    grapes=grapes_list,
                    rating=float(wine['avg_rating']) if 'avg_rating' in wine else None,
                    similarity=float(wine['Similarity']) if 'Similarity' in wine else None
                ))

        # Create response object with both recommendations and extracted info
        wine_info = WineInfo(
            wine_type=extracted_info.get('wine_type'),
            grape_varieties=extracted_info.get('grape_varieties'),
            body=extracted_info.get('body'),
            acidity=extracted_info.get('acidity'),
            country=extracted_info.get('country'),
            region=extracted_info.get('region'),
            ABV=extracted_info.get('ABV'),
            extraction_successful=extracted_info.get('extraction_successful', False),
            error=extracted_info.get('error')
        )

        return WineRecommendationsResponse(
            recommendations=result,
            extracted_info=wine_info
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
