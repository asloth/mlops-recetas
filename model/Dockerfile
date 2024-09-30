# Base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy your requirements.txt and install dependencies
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y git
# Install GCC and other build tools
RUN apt-get update && apt-get install -y build-essential gcc
RUN pip install -r requirements.txt
