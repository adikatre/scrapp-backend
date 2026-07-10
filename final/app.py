import os

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import io
from flask_cors import CORS
from dotenv import load_dotenv

from categories import build_coco_to_bin, default_route
from openai_classifier import classify_waste, classification_to_response_fields

load_dotenv()

USE_YOLO = os.getenv("USE_YOLO", "false").lower() in {"1", "true", "yes"}

model = None
COCO_TO_BIN = {}

if USE_YOLO:
    from ultralytics import YOLO

    # Load YOLO model once at startup
    model = YOLO("yolov8x.pt")

    # Build class-to-bin mapping
    COCO_TO_BIN = build_coco_to_bin(model.names)

app = Flask(__name__)
CORS(app)


def run_yolo_detection(img):
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

        bin_totals[route] = bin_totals.get(route, 0) + 1

        x1, y1, x2, y2 = map(float, b.xyxy[0])

        detections.append({
            "class_name": name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
            "route": route
        })
        objects.append(name)

    return objects, detections, bin_totals


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file field provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    user_text = request.form.get("text", "").strip() or None

    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if USE_YOLO:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        objects, detections, bin_totals = run_yolo_detection(img)
    else:
        objects, detections, bin_totals = [], [], {}

    classification = classify_waste(
        pil_img=pil_img,
        detections=detections,
        user_text=user_text,
    )
    classifier_fields = classification_to_response_fields(classification)

    response = {
        "objects": objects,
        "detections": detections,
        "bin_totals": bin_totals,
        **classifier_fields,
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
