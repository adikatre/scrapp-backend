from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
from flask_cors import CORS


#download and load model once server starts.
model = YOLO("yolov8x.pt")

recycle = {
    "bottle", "wine glass", "cup", "bowl",
    "fork", "knife", "spoon",
    "vase", "chair", "bench",
    "laptop", "mouse", "keyboard", "cell phone",
}

compost = {
    "banana", "apple", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake", "sandwich", "hot dog",
}

store_dropoff = {
    "handbag", "backpack", "umbrella" # these usually contain flimsy plastics
}

default_route = "Landfill / Donate / Check rules"

COCO_TO_BIN = {}
for id, name in model.names.items():
    if name in recycle:
        COCO_TO_BIN[name] = "Recycle"
    elif name in compost:
        COCO_TO_BIN[name] = "Compost"
    elif name in store_dropoff:
        COCO_TO_BIN[name] = "Store Drop-off"
    else:
        COCO_TO_BIN[name] = default_route
        
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file field provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400
    
    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    results = model.predict([img], conf=0.30, verbose=False)
    r = results[0]
    
    detections = []
    bin_totals = {}
    
    #handle bounding boxes of detected objects
    
    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf   = float(b.conf[0])
        name   = r.names[cls_id]
        route  = COCO_TO_BIN.get(name, default_route)

        # Aggregate per-bin count
        bin_totals[route] = bin_totals.get(route, 0) + 1

        # Bounding box (xyxy in image pixels)
        x1, y1, x2, y2 = map(float, b.xyxy[0])

        detections.append({
            "class_name": name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "route": route
    })
        
    return jsonify({
        "detections": detections,
        "bin_totals": bin_totals
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")