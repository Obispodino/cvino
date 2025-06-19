# 1. ✅ Use a base image
FROM python:3.11-slim

# 2. ✅ Set working directory
WORKDIR /API

# 3. ✅ Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache

# 4. ✅ Copy your actual code
COPY cv_functions cv_functions
COPY API API
COPY raw_data raw_data
COPY transformers transformers
COPY models models
COPY images images
COPY interface interface

# 5. ✅ Define the entrypoint to run the API

CMD uvicorn fast_api.app:app --host 0.0.0.0 --port $PORT
