# 1. ✅ Use a base image
FROM python:3.11-slim

# 2. ✅ Set working directory
WORKDIR /app

# 3. ✅ Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. ✅ Copy your actual code
COPY cv_functions cv_functions
COPY API API
COPY raw_data raw_data
COPY models models

# 5. ✅ Define the entrypoint to run the API
CMD ["uvicorn", "API.fast:app", "--host", "0.0.0.0", "--port", "8080"]
