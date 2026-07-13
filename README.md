# Scrapp Backend

Flask API that powers [Scrapp](https://github.com/adikatre/scrapp) waste classification. Accepts an image (and optional text note), runs GPT-4o-mini vision, and returns structured disposal guidance.

**Frontend:** [github.com/adikatre/scrapp](https://github.com/adikatre/scrapp)

## Overview

- `POST /predict` — classify a waste item from a photo
- **Primary classifier:** GPT-4o-mini vision ([openai_classifier.py](final/openai_classifier.py)) — structured JSON with item name, material, disposal route, confidence, caveats, and Places search queries
- **Optional:** YOLOv8 object detection behind `USE_YOLO=true` (off by default; requires extra dependencies and ~2–4 GB RAM; not used in production)
- **Auth:** every request needs `Authorization: Bearer $BACKEND_API_KEY` (see [Authentication](#authentication))

## Local Development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # fill in OPENAI_API_KEY and BACKEND_API_KEY
python final/app.py        # listens on :5000
```

The frontend expects the backend at `http://localhost:5000` by default, and must be given the same `BACKEND_API_KEY` — a mismatch turns every scan into a `401`.

## Authentication

The browser never calls this API directly; the Next.js server action does. So CORS and `Origin` checks don't authenticate anything — a shared secret does. A `before_request` hook ([app.py](final/app.py)) rejects any request whose `Authorization: Bearer <key>` doesn't match `BACKEND_API_KEY` (constant-time compare), including when the variable is unset — so an unconfigured server refuses everything rather than serving your OpenAI quota to the internet. `OPTIONS` preflights are exempt.

Generate a key with:

```bash
openssl rand -hex 32
```

Set the identical value on the backend host and in the frontend's environment.

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | GPT-4o-mini vision API calls |
| `BACKEND_API_KEY` | Yes | — | Shared bearer secret the frontend must send; unset means every request `401`s |
| `USE_YOLO` | No | `false` | Enable YOLOv8 detection (needs `ultralytics`, `opencv-python-headless`, and significant RAM) |
| `CORS_ORIGINS` | No | — | Comma-separated extra allowed origins, added to the production defaults below |
| `OPENAI_ENABLED` | No | `true` | Set false to skip the API call and return the heuristic fallback |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Vision model id |
| `OPENAI_IMAGE_DETAIL` | No | `high` | `high` or `low` — `low` is cheaper and coarser |
| `OPENAI_TIMEOUT_SECONDS` | No | `20` | Per-request timeout before falling back |
| `OPENAI_MAX_IMAGE_SIZE` | No | `1280` | Longest edge, in px, images are resized to |
| `OPENAI_JPEG_QUALITY` | No | `85` | JPEG quality of the re-encoded upload |

See [.env.example](.env.example) for a template.

## Tests

```bash
USE_YOLO=true python -m pytest
```

`USE_YOLO=true` is required for the **full** suite: `app.py` imports `cv2`/`numpy` only when the flag is on at import time, so the two YOLO integration tests raise `NameError` without it (30 pass, 2 fail). It also means those two tests need `ultralytics` and `opencv-python-headless` installed — neither is in [requirements.txt](requirements.txt), since production doesn't run YOLO. Plain `python -m pytest` exercises everything else, including the OpenAI classifier and category mapping, against mocks — no API key needed.

## API Reference

### `POST /predict`

Classify a waste item from an image.

**Request:** `multipart/form-data`

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <BACKEND_API_KEY>` |

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
| `401` | Missing, malformed, or wrong `Authorization: Bearer` token — or `BACKEND_API_KEY` is unset on the server |
| `400` | Missing `file` field or empty filename |

## Production Deployment

### Docker

The [Dockerfile](Dockerfile) builds a slim Python 3.12 image and runs gunicorn:

```bash
docker build -t scrapp-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... -e BACKEND_API_KEY=... scrapp-backend
```

Gunicorn binds to `$PORT` (default `8000`) with 2 workers and a 60-second timeout.

### Railway / Render

1. Connect this repo
2. Set `OPENAI_API_KEY` and `BACKEND_API_KEY` in environment variables
3. Keep `USE_YOLO=false` (default)
4. Deploy — the host builds from the Dockerfile automatically
5. Point the frontend's `NEXT_PRIVATE_BACKEND_URL` at the deployed URL and give it the same `BACKEND_API_KEY`

CORS is enabled for `https://scrapp.app`, `https://www.scrapp.app`, and `https://scrapp-sd.vercel.app`. Set `CORS_ORIGINS` (comma-separated) to allow additional origins during development. Note that CORS is not the security boundary here — `BACKEND_API_KEY` is.

## Classifier Details

- **Implementation:** [final/openai_classifier.py](final/openai_classifier.py)
- **Model:** `gpt-4o-mini` with structured JSON output
- **Image handling:** Resized to max 1280px, JPEG quality 85, `detail: high` (all tunable — see the `OPENAI_*` variables above)
- **Cost:** Approximately $0.001–0.003 per scan at high detail
- **Fallback:** If the API key is missing, the call times out, or `OPENAI_ENABLED=false`, the classifier returns a heuristic result with `"fallback_used": true` and `"model": "fallback"` instead of failing the request

To localize disposal rules, edit the system prompt in `openai_classifier.py` with your city or county's actual recycling guidelines.

## Made By

- Frontend: [SlushEE0](https://github.com/slushee0) — [repo](https://github.com/adikatre/scrapp)
- Backend & AI: [adikatre](https://github.com/adikatre) — [repo](https://github.com/adikatre/scrapp-backend)
