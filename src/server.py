from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
from contextlib import asynccontextmanager
import json

model = None

def get_model():
    global model
    if model is None:
       model = YOLO("yolov8n.pt")
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Hello from Blueberry!"}

@app.get("/predict")
def predict(image_url: str):
    try:
        results = get_model()(image_url)
        return {
            "results": [json.loads(r.to_json()) for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
