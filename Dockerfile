# Use a sleek Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system utilities for NLP models
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset and application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn with high-performance settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
