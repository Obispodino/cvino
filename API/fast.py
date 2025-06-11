# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# CORS middleware (to allow access from frontend apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load the model if it exists
model = None
model_path = "model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
else:
    print("⚠️ Warning: model.joblib not found. Predictions will not work.")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Hi, the API is running!"}

# GET endpoint for quick testing
@app.get("/predict")
def get_predict(input_one: float, input_two: float):
    if not model:
        return {"error": "Model not loaded."}
    try:
        prediction = model.predict([[input_one, input_two]])[0]
        return {
            "prediction": int(prediction),
            "inputs": {
                "input_one": input_one,
                "input_two": input_two
            }
        }
    except Exception as e:
        return {"error": str(e)}

# POST endpoint for JSON-based predictions
class PredictionRequest(BaseModel):
    input_one: float
    input_two: float

@app.post("/predict")
def predict_post(request: PredictionRequest):
    if not model:
        return {"error": "Model not loaded."}
    try:
        prediction = model.predict([[request.input_one, request.input_two]])[0]
        return {
            "prediction": int(prediction),
            "inputs": request.dict()
        }
    except Exception as e:
        return {"error": str(e)}
