from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
from flask_cors import CORS

# Import category mappings
from categories import build_coco_to_bin, default_route

# Load YOLO model once at startup
model = YOLO("yolov8n.pt")

# Build class-to-bin mapping
COCO_TO_BIN = build_coco_to_bin(model.names)

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

    objects = []
    detections = []
    bin_totals = {}

    for b in r.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = r.names[cls_id]
        route = COCO_TO_BIN.get(name, default_route)

        # Aggregate per-bin count
        bin_totals[route] = bin_totals.get(route, 0) + 1

        # Bounding box
        x1, y1, x2, y2 = map(float, b.xyxy[0])

        detections.append({
            "class_name": name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "route": route
        })
        objects.append(name)

    return jsonify({
        "objects": objects,
        "detections": detections,
        "bin_totals": bin_totals
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
