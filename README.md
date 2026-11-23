# Scrapp Backend

Flask API that powers [Scrapp](https://github.com/adikatre/scrapp) waste classification. Accepts a photo, a written item description, or both. A vision model identifies the item and returns preparation steps and a disposal route for City of San Diego households.

**Frontend:** [github.com/adikatre/scrapp](https://github.com/adikatre/scrapp)

## Overview

- `POST /predict` returns item names, materials, disposal routes, bins, confidence estimates, caveats and search terms.
- The server selects Adaption when `ADAPTION_API_KEY` is present. Existing OpenAI-only deployments continue to use GPT-4o-mini.
- **Optional:** YOLOv8 object detection behind `USE_YOLO=true` (off by default; requires extra dependencies and ~2–4 GB RAM; not used in production)
- **Auth:** every request needs `Authorization: Bearer $BACKEND_API_KEY` (see [Authentication](#authentication))

## Local Development

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp -n .env.example .env    # preserves existing keys
# Set ADAPTION_API_KEY and BACKEND_API_KEY in .env
python final/app.py        # listens on :5000
```

The frontend expects the backend at `http://localhost:5000` by default, and must be given the same `BACKEND_API_KEY`, a mismatch turns every scan into a `401`.

## Authentication

The browser never calls this API directly; the Next.js server action does. So CORS and `Origin` checks don't authenticate anything, a shared secret does. A `before_request` hook ([app.py](final/app.py)) rejects any request whose `Authorization: Bearer <key>` doesn't match `BACKEND_API_KEY` (constant-time compare), including when the variable is unset, so an unconfigured server refuses everything rather than using a provider's quota for unauthenticated scans. `OPTIONS` preflights are exempt.

Generate a key with:

```bash
openssl rand -hex 32
```

Set the identical value on the backend host and in the frontend's environment.

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ADAPTION_API_KEY` | For Adaption | — | Key for the hosted Responses API. Takes priority when both provider keys are set |
| `ADAPTION_MODEL` | No | `qwen3.5-397b` | Hosted model with image input support |
| `ADAPTION_TIMEOUT_SECONDS` | No | `20` | Adaption request timeout |
| `ADAPTION_ENABLED` | No | `true` | Set to false to exercise the fallback on an Adaption installation |
| `OPENAI_API_KEY` | For OpenAI only | — | Alternative provider when the Adaption key is absent |
| `BACKEND_API_KEY` | Yes |, | Shared bearer secret the frontend must send; unset means every request `401`s |
| `USE_YOLO` | No | `false` | Enable YOLOv8 detection (needs `ultralytics`, `opencv-python-headless`, and significant RAM) |
| `CORS_ORIGINS` | No |, | Comma-separated extra allowed origins, added to the production defaults below |
| `OPENAI_ENABLED` | No | `true` | Set false to skip the API call and return the heuristic fallback |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Vision model id |
| `OPENAI_IMAGE_DETAIL` | No | `high` | `high` or `low`, `low` is cheaper and coarser |
| `OPENAI_TIMEOUT_SECONDS` | No | `20` | Per-request timeout before falling back |
| `OPENAI_MAX_IMAGE_SIZE` | No | `1280` | Longest edge, in px, images are resized to |
| `OPENAI_JPEG_QUALITY` | No | `85` | JPEG quality of the re-encoded upload |

See [.env.example](.env.example) for a template.

## Tests

```bash
USE_YOLO=false python -m pytest
```

The suite exercises photo-only, text-only and combined requests, optional YOLO behavior, provider failures, input validation and bearer authentication. It uses mocked providers and requires no live API credentials or model weights.

## API Reference

### `POST /predict`

Classify a waste item from an image, an item description, or both. Provide at least one of the following fields.

**Request:** `multipart/form-data`

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <BACKEND_API_KEY>` |

| Field | Required | Description |
|---|---|---|
| `file` | If text is empty | Decodable photo such as JPEG, PNG or WebP, up to 32 megapixels. Total request limit is 12 MiB |
| `text` | If file is absent | Item description or context note, up to 1,000 characters after trimming |

**Response:** `200 OK`, JSON body:

```json
{
  "items": [
    {
      "name": "Plastic yogurt cup",
      "material": "rigid plastic",
      "route": "Recycle",
      "bin": "Blue Bin (Recycling)",
      "confidence": 0.9,
      "caveats": "Empty and dry the container first.",
      "search_queries": []
    }
  ],
  "text": "Place the empty, dry yogurt cup loose in your blue recycling cart.",
  "classifier": {
    "model": "qwen3.5-397b",
    "source": "adaption",
    "basis": "image",
    "fallback_used": false
  },
  "objects": [],
  "detections": [],
  "bin_totals": {}
}
```

`objects`, `detections`, and `bin_totals` are populated only when `USE_YOLO=true`. The response shape matches the frontend's [`PredictionResult`](https://github.com/adikatre/scrapp/blob/main/src/lib/types.ts) type.

`classifier.basis` records what informed the result. Model results use `image`, `note` or `image_and_note`. A provider outage or invalid model response sets `fallback_used` to true. Available YOLO labels then use `detections`. An empty fallback returns no items and uses `source: "none"` and `basis: "none"`. Successful model responses may also have an empty item list, with guidance asking for more information.

**Valid disposal routes:** Recycle, Compost, E-Waste, Bulky Items (Donate), Hazardous Waste, Single-Use Items, City Infrastructure, Living Things, General Trash, Landfill / Donate / Check rules

**Errors:**

| Status | Condition |
|---|---|
| `401` | Missing, malformed, or wrong `Authorization: Bearer` token, or `BACKEND_API_KEY` is unset on the server |
| `400` | Empty input, unreadable image, or description over 1,000 characters |
| `413` | Request exceeds 12 MiB or photo exceeds 32 megapixels |

## Production Deployment

### Docker

The [Dockerfile](Dockerfile) builds a slim Python 3.12 image and runs gunicorn:

```bash
docker build -t scrapp-backend .
docker run -p 8000:8000 --env-file .env scrapp-backend
```

Gunicorn binds to `$PORT` (default `8000`) with 2 workers and a 60-second timeout.

### Railway / Render

1. Connect this repo
2. Set `ADAPTION_API_KEY` (or the alternative `OPENAI_API_KEY`) and `BACKEND_API_KEY` in environment variables
3. Keep `USE_YOLO=false` (default)
4. Deploy, the host builds from the Dockerfile automatically
5. Point the frontend's `NEXT_PRIVATE_BACKEND_URL` at the deployed URL and give it the same `BACKEND_API_KEY`

CORS is enabled for `https://scrapp.app`, `https://www.scrapp.app`, and `https://scrapp-sd.vercel.app`. Set `CORS_ORIGINS` (comma-separated) to allow additional origins during development. Note that CORS is not the security boundary here, `BACKEND_API_KEY` is.

## Classifier Details

### Image and description classification

[The classifier](final/openai_classifier.py) sends one provider request per scan. Adaption receives a message at `POST https://api.prod.adaptionlabs.ai/api/v1/responses`. A photo is a resized JPEG data URL in an `input_image` part. The optional note and YOLO hints are sent as text. The request sets `store: false`, uses no tools and caps output at 900 tokens. The default Qwen 3.5 model supports both text and image input. See the [referenced endpoint guide](https://github.com/anyhows/SwarmAlignment26/blob/main/docs/adaption-llm-endpoints.md), [Adaption image input reference](https://docs.adaptionlabs.ai/inference-stacks/generate-a-model-response/) and [hosted model list](https://docs.adaptionlabs.ai/inference-stacks/models/).

OpenAI-only installations use Chat Completions with a JSON schema. Both providers share the same item validation, route and bin vocabulary, camera orientation correction, image resizing and response fields. The client never receives provider keys. Provider errors use the existing detection fallback, with no automatic retries or second paid provider call.

The prompt includes the City's published curbside sorting defaults and asks the model to identify material, condition, and any uncertainty. The frontend pairs the item with local program information. [City of San Diego What Goes Where guide](https://www.sandiego.gov/sites/default/files/what_goes_where_guide_english.pdf) · [City household hazardous waste programs](https://www.sandiego.gov/environmental-services/ep/hazardous)

### YOLOv8 detection and disposal mapping

The optional YOLO path identifies individual objects in a photo and maps their labels to disposal routes. The current implementation loads the pretrained **YOLOv8x** detector (`yolov8x.pt`); the waste-specific behavior comes from hand-written category mappings, rather than fine-tuned model weights. This repository does not include a custom waste checkpoint or a fine-tuning pipeline.

- **Implementation:** [final/app.py](final/app.py), with class-to-route rules in [final/categories.py](final/categories.py)
- **Detection:** The model loads once at startup. Each uploaded photo is decoded as a PIL RGB image and passed to YOLO with a confidence threshold of `0.30`. Each detection includes a class name, confidence score, and bounding box in image pixel coordinates.
- **Disposal mapping:** COCO object labels are mapped to routes such as Recycle (`bottle`), Compost (`banana`), and E-Waste (`laptop`). Unmapped labels use `Landfill / Donate / Check rules`. These rules use object labels, so they cannot reliably distinguish materials, contamination, or local recycling exceptions.
- **Response:** `objects` lists detected labels, `detections` includes boxes and suggested routes, and `bin_totals` counts detections per route.
- The selected vision provider receives waste detections as hints and can correct them using the photo or description. If the provider is unavailable, the same detections supply the fallback items.

To enable YOLO locally, install the additional dependencies and set the flag before starting the backend:

```bash
pip install ultralytics opencv-python-headless numpy
USE_YOLO=true python final/app.py
```

YOLO is off by default and is not used in the current production configuration because of its additional memory requirements.

## Made By

- [SlushEE0](https://github.com/slushee0), [repo](https://github.com/adikatre/scrapp)
- [adikatre](https://github.com/adikatre), [repo](https://github.com/adikatre/scrapp-backend)
