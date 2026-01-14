from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from contextlib import asynccontextmanager
import json

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO("yolo11n.pt")  # loads once on startup
    yield
    model = None

app = FastAPI(lifespan=lifespan)

@app.get("/hello")
def home():
    return {"message": "Hello from Render!"}

@app.get("/predict")
def predict(image_url: str):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = model(image_url)
    return {
        "results": [json.loads(r.to_json()) for r in results]
    }
