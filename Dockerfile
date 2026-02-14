# Use a stable, lightweight Python environment
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies (required for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY requriements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requriements.txt

# Copy the entire project code into the container
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Note: We use docker-compose to decide which command to run
# This Dockerfile acts as the base image for both services.
