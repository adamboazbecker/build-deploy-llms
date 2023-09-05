FROM python:3.9.7-slim

RUN apt-get update && apt-get install -y build-essential

RUN mkdir /app

# Your Dockerfile instructions here

# These are the relevant files but we should be mounting them instead...
#COPY docs_sample app/docs_sample
#COPY src app/src
#COPY .env.example app/.env.example
COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt
