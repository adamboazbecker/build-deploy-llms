FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential

RUN mkdir /app

# Your Dockerfile instructions here

# These are the relevant files but we should be mounting them instead...
COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
