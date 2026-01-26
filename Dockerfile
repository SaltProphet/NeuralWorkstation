# FORGE v1 - Neural Audio Workstation
# Dockerfile for containerized deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel to ensure proper binary package handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with preference for binary packages
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application files
COPY forgev1.py .
COPY app.py .
COPY README.md .
COPY LICENSE .

# Create necessary directories
RUN mkdir -p runs cache config checkpoint feedback output/stems output/loops output/chops output/midi output/drums output/videos

# Expose Gradio default port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app.py"]
