# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Run the FastAPI app via uvicorn
CMD ["uvicorn", "API.fast:app", "--host", "0.0.0.0", "--port", "8000"]
