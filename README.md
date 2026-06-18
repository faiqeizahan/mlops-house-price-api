# House Price Prediction API

A simple MLOps project using:

* Scikit-Learn
* FastAPI
* Docker
* Swagger UI

## Features

* Train machine learning model
* Save model as model.pkl
* Serve predictions through FastAPI
* Run inside Docker container

## Run Locally

```bash
python train.py
uvicorn app.main:app --reload
```

## Run with Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## API Documentation

Open:

http://localhost:8000/docs
