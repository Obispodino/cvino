FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY API/ ./API/
COPY cv_functions/ ./cv_functions/
COPY trained_model_Dino.pkl .

# Expose FastAPI's port
EXPOSE 8080

# Run FastAPI with uvicorn
CMD ["uvicorn", "API.fast:app", "--host", "0.0.0.0", "--port", "8080"]
