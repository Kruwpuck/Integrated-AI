# Use official Python runtime as a parent image
# python 3.9 slim is a good balance between size and compatibility
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are required for cv2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
# Note: This assumes the "model DL" folder and models are in the build context
COPY . .

# Create output directories
RUN mkdir -p output_api/images output_api/json

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
# We use app_docker:app
CMD ["uvicorn", "app_docker:app", "--host", "0.0.0.0", "--port", "8000"]
