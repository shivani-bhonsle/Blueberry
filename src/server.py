from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import json

app = FastAPI()

# Load the model once on startup
try:
    model = YOLO("yolo11m.pt")  # Make sure yolo11m.pt is in the repo
except Exception as e:
    print("Error loading YOLO model:", e)
    model = None

@app.get("/hello")
def home():
    return {"message": "Hello from Render!"}

@app.get("/predict")
def predict(image_url: str):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = model(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    output = []
    for result in results:
        # Only return JSON results, no GUI calls
        data = json.loads(result.to_json())
        output.append(data)
        
    return {"results": output}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
