# Use a slim Python 3.10 image to keep the container size manageable
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_NAME="all-MiniLM-L6-v2" \
    HOME=/home/user

# Create a non-root user (Hugging Face requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR $HOME/app

# Copy requirements and install dependencies
# We do this first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD ML MODEL ---
# This ensures the model is baked into the image and doesn't download on every start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$MODEL_NAME')"

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860 (Default for Hugging Face Spaces)
EXPOSE 7860

# Start the application
# We use 0.0.0.0 to allow external traffic to the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
