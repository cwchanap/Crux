# Base image with Python and slim Debian
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for audio processing (librosa/soundfile) and general tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (leverage Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create temp directories used by the app (optional)
RUN mkdir -p temp_uploads temp_downloads

# Expose API port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
