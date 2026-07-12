# Scrapp Backend

Flask API that powers [Scrapp](https://github.com/adikatre/scrapp-backend) waste classification. Accepts an image (and optional text note), runs GPT-4o-mini vision, and returns structured disposal guidance.

**Frontend:** [github.com/adikatre/scrapp](https://github.com/adikatre/scrapp)

## Overview

- `POST /predict` — classify a waste item from a photo
- **Primary classifier:** GPT-4o-mini vision ([openai_classifier.py](final/openai_classifier.py)) — structured JSON with item name, material, disposal route, confidence, caveats, and Places search queries
- **Optional:** YOLOv8 object detection behind `USE_YOLO=true` (off by default; requires extra dependencies and ~2–4 GB RAM; not used in production)

## Local Development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python final/app.py        # listens on :5000
```

The frontend expects the backend at `http://localhost:5000` by default.

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | GPT-4o-mini vision API calls |
| `USE_YOLO` | No | `false` | Enable YOLOv8 detection (needs `ultralytics`, `opencv-python-headless`, and significant RAM) |

See [.env.example](.env.example) for a template.

## API Reference

### `POST /predict`

Classify a waste item from an image.

**Request:** `multipart/form-data`

| Field | Required | Description |
|---|---|---|
| `file` | Yes | Image file (JPEG, PNG, etc.) |
| `text` | No | Optional user context note |

**Response:** `200 OK` — JSON body:

```json
{
  "items": [
    {
      "name": "Plastic yogurt cup",
      "material": "#5 polypropylene",
      "route": "Recycle",
      "confidence": 0.9,
      "caveats": "Rinse before recycling; check local #5 acceptance",
      "search_queries": ["plastic recycling center", "yogurt container recycling"]
    }
  ],
  "text": "This appears to be a #5 plastic yogurt cup. Rinse it and check whether your curbside program accepts #5 plastics.",
  "classifier": {
    "model": "gpt-4o-mini",
    "source": "openai",
    "fallback_used": false
  },
  "objects": [],
  "detections": [],
  "bin_totals": {}
}
```

`objects`, `detections`, and `bin_totals` are populated only when `USE_YOLO=true`. The response shape matches the frontend's [`PredictionResult`](https://github.com/adikatre/scrapp/blob/main/src/lib/types.ts) type.

**Valid disposal routes:** Recycle, Compost, E-Waste, Bulky Items (Donate), Hazardous Waste, Single-Use Items, City Infrastructure, Living Things, General Trash, Landfill / Donate / Check rules

**Errors:**

| Status | Condition |
|---|---|
| `400` | Missing `file` field or empty filename |

## Production Deployment

### Docker

The [Dockerfile](Dockerfile) builds a slim Python 3.12 image and runs gunicorn:

```bash
docker build -t scrapp-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... scrapp-backend
```

Gunicorn binds to `$PORT` (default `8000`) with 2 workers and a 60-second timeout.

### Render / Railway

1. Connect this repo
2. Set `OPENAI_API_KEY` in environment variables
3. Keep `USE_YOLO=false` (default)
4. Deploy — Render builds from the Dockerfile automatically

CORS is enabled for `https://scrapp.app`, `https://www.scrapp.app`, and `https://scrapp-sd.vercel.app`. Set `CORS_ORIGINS` (comma-separated) to allow additional origins during development.

## Classifier Details

- **Implementation:** [final/openai_classifier.py](final/openai_classifier.py)
- **Model:** `gpt-4o-mini` with structured JSON output
- **Image handling:** Resized to max 1280px, JPEG quality 85, `detail: high`
- **Cost:** Approximately $0.001–0.003 per scan at high detail

To localize disposal rules, edit the system prompt in `openai_classifier.py` with your city or county's actual recycling guidelines.

## Made By

- Frontend: [SlushEE0](https://github.com/slushee0) — [repo](https://github.com/adikatre/scrapp)
- Backend & AI: [adikatre](https://github.com/adikatre) — [repo](https://github.com/adikatre/scrapp-backend)
