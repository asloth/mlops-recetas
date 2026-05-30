# Base image
FROM python:3.12.11-slim

# Set the working directory
WORKDIR /app

# Copy your requirements.txt and install dependencies
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
