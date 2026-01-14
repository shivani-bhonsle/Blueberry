from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
import json

app = FastAPI()

# Load a model
# using 'yolo11n.pt' as in the original code (assuming it exists or will be downloaded)
model = YOLO("yolo11m.pt")

@app.get("/hello")
def home():
    return "Hello from Render!"

@app.get("/predict")
def predict(image_url: str):
    """
    Run inference on the provided image URL.
    """
    print(f"Processing URL: {image_url}")
    
    # Run inference
    # The model() call returns a list of Results objects
    results = model(image_url)
    
    # Process results
    output = []
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()
        result.save()
        # result.to_json() returns a JSON string representing the detections
        json_str = result.to_json()
        # Parse it back to a python object so FastAPI can serialize it properly in the response
        data = json.loads(json_str)
        print(data)
        output.append(data)
        
    return {"results": output}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)