FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only what's needed for inference
COPY API/ ./API/
COPY cv_functions/ ./cv_functions/
COPY transformers/ ./transformers/

# Minimal model and data files
COPY models/trained_model.pkl ./models/
COPY models/preprocessor.pkl ./models/
COPY raw_data/wine_metadata.csv ./raw_data/
COPY raw_data/geocoding_cache.pkl ./raw_data/

EXPOSE 8080

CMD ["uvicorn", "API.fast:app", "--host", "0.0.0.0", "--port", "8080"]
