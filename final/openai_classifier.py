"""OpenAI vision classifier for waste disposal guidance."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from categories import default_route, is_non_waste_route, normalize_route

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_DETAIL = "high"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_IMAGE_SIZE = 1280
DEFAULT_JPEG_QUALITY = 85

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "material": {"type": "string"},
                    "route": {"type": "string"},
                    "confidence": {"type": "number"},
                    "caveats": {"type": "string"},
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "name",
                    "material",
                    "route",
                    "confidence",
                    "caveats",
                    "search_queries",
                ],
                "additionalProperties": False,
            },
        },
        "guidance": {"type": "string"},
    },
    "required": ["items", "guidance"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a waste disposal expert helping users sort items correctly.

Given an image (and optional YOLO detections or user context), identify visible waste items and recommend disposal routes.

Valid disposal routes (use exactly one of these strings):
- Recycle
- Compost
- E-Waste
- Bulky Items (Donate)
- Hazardous Waste
- Single-Use Items
- City Infrastructure
- Living Things
- General Trash
- Landfill / Donate / Check rules

Rules:
- Focus on the held or foreground waste item the user is showing, not people, furniture, or background scenery.
- Do not classify people as waste. Ignore Living Things unless the user is clearly asking about an animal.
- Small electronics accessories (USB cables, chargers, cords, earbuds, power adapters, HDMI cables) are E-Waste.
- Be conservative for hazardous items; prefer "Hazardous Waste" or "Landfill / Donate / Check rules" when unsure.
- Consider material (plastic, glass, metal, organic, electronic) when choosing routes.
- If YOLO detections are provided, use them as hints but correct mistakes when the image shows something different.
- Keep guidance concise (2-4 sentences) and practical for a household user.
- confidence is 0.0 to 1.0 for your classification certainty.
- search_queries: 2-4 short Google Places text-search queries that find real nearby drop-off
  points for THIS specific item and material — never generic category queries. Include the
  item-specific facility type (e.g. "battery recycling drop-off", "glass bottle redemption
  center", "food scrap compost drop-off") AND names of retail chains or public places known
  to accept it (e.g. batteries -> "Ralphs", "Home Depot", "public library"; paint ->
  "Sherwin-Williams"; ink cartridges -> "Staples"; clothes -> "Goodwill donation center";
  plastic bags -> "grocery store plastic bag recycling"). Provide queries for every route
  with a physical drop-off (Recycle, Compost, E-Waste, Hazardous Waste, Bulky Items
  (Donate), Landfill / Donate / Check rules). Do not include any location or "near me"
  wording. Use an empty array only for Single-Use Items, General Trash, City
  Infrastructure, and Living Things.
"""

PERSON_LIKE_NAMES = frozenset({
    "person",
    "human",
    "man",
    "woman",
    "child",
    "people",
})


@dataclass
class ClassifierConfig:
    enabled: bool
    model: str
    image_detail: str
    timeout_seconds: float
    max_image_size: int
    jpeg_quality: int


@dataclass
class ClassificationResult:
    items: list[dict[str, Any]]
    guidance: str
    model: str
    source: str
    fallback_used: bool


def get_classifier_config() -> ClassifierConfig:
    return ClassifierConfig(
        enabled=os.getenv("OPENAI_ENABLED", "true").lower() in {"1", "true", "yes"},
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        image_detail=os.getenv("OPENAI_IMAGE_DETAIL", DEFAULT_IMAGE_DETAIL),
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
        max_image_size=int(os.getenv("OPENAI_MAX_IMAGE_SIZE", str(DEFAULT_MAX_IMAGE_SIZE))),
        jpeg_quality=int(os.getenv("OPENAI_JPEG_QUALITY", str(DEFAULT_JPEG_QUALITY))),
    )


def prepare_image_data_url(pil_img: Image.Image, config: ClassifierConfig) -> str:
    """Resize and compress an image for cost-efficient OpenAI vision input."""
    img = pil_img.convert("RGB")
    img.thumbnail((config.max_image_size, config.max_image_size), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=config.jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _is_waste_item(item: dict[str, Any]) -> bool:
    route = str(item.get("route", ""))
    if is_non_waste_route(route):
        return False

    name = str(item.get("name", "")).strip().lower()
    if name in PERSON_LIKE_NAMES:
        return False

    return True


def _sort_items_by_relevance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            0 if _is_waste_item(item) else 1,
            -float(item.get("confidence", 0.0)),
        ),
    )


def _waste_detections_for_hints(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        detection
        for detection in detections
        if not is_non_waste_route(str(detection.get("route", "")))
    ]


def _build_user_content(
    image_data_url: str,
    detections: list[dict[str, Any]],
    user_text: str | None,
    config: ClassifierConfig,
) -> list[dict[str, Any]]:
    detection_summary = [
        {
            "class_name": d.get("class_name"),
            "confidence": d.get("confidence"),
            "route": d.get("route"),
            "bbox": d.get("bbox"),
        }
        for d in _waste_detections_for_hints(detections)
    ]

    payload = {
        "yolo_detections": detection_summary,
        "user_context": user_text or "",
    }

    return [
        {
            "type": "text",
            "text": (
                "Classify the waste items in this image. "
                f"Context JSON: {json.dumps(payload, ensure_ascii=True)}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url,
                "detail": config.image_detail,
            },
        },
    ]


def _normalize_search_queries(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    queries = []
    for entry in raw:
        query = str(entry).strip()
        if query:
            queries.append(query[:60])
        if len(queries) >= 4:
            break
    return queries


def _parse_response_content(content: str) -> dict[str, Any]:
    parsed = json.loads(content)

    items = parsed.get("items", [])
    if not isinstance(items, list):
        raise ValueError("items must be a list")

    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized_items.append({
            "name": str(item.get("name", "unknown item")),
            "material": str(item.get("material", "unknown")),
            "route": normalize_route(str(item.get("route", default_route))),
            "confidence": float(item.get("confidence", 0.5)),
            "caveats": str(item.get("caveats", "")),
            "search_queries": _normalize_search_queries(item.get("search_queries")),
        })

    guidance = str(parsed.get("guidance", "")).strip()

    return {
        "items": _sort_items_by_relevance(normalized_items),
        "guidance": guidance,
    }


def _fallback_result(detections: list[dict[str, Any]], reason: str) -> ClassificationResult:
    waste_detections = _waste_detections_for_hints(detections)

    items = []
    for detection in waste_detections:
        items.append({
            "name": detection.get("class_name", "unknown"),
            "material": "unknown",
            "route": detection.get("route", default_route),
            "confidence": float(detection.get("confidence", 0.0)),
            "caveats": reason,
            "search_queries": [],
        })

    if not items:
        guidance = (
            "No items were confidently detected. "
            "Please try a clearer photo or check local disposal rules."
        )
    else:
        routes = sorted({item["route"] for item in items})
        guidance = (
            "Classification used detection-only fallback. "
            f"Suggested routes: {', '.join(routes)}."
        )

    return ClassificationResult(
        items=items,
        guidance=guidance,
        model="fallback",
        source="yolo",
        fallback_used=True,
    )


def classify_waste(
    pil_img: Image.Image,
    detections: list[dict[str, Any]],
    user_text: str | None = None,
    config: ClassifierConfig | None = None,
    client: OpenAI | None = None,
) -> ClassificationResult:
    """Run OpenAI vision classification with safe fallback to YOLO-only output."""
    config = config or get_classifier_config()

    if not config.enabled:
        return _fallback_result(detections, "OpenAI classifier disabled")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; using YOLO fallback")
        return _fallback_result(detections, "OpenAI API key not configured")

    try:
        image_data_url = prepare_image_data_url(pil_img, config)
        openai_client = client or OpenAI(api_key=api_key, timeout=config.timeout_seconds)

        response = openai_client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_content(
                        image_data_url,
                        detections,
                        user_text,
                        config,
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "waste_classification",
                    "strict": True,
                    "schema": CLASSIFICATION_SCHEMA,
                },
            },
            temperature=0.2,
            max_tokens=700,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")

        parsed = _parse_response_content(content)

        # Ensure routes remain valid even if model drifts outside schema constraints.
        for item in parsed["items"]:
            item["route"] = normalize_route(item["route"])

        return ClassificationResult(
            items=parsed["items"],
            guidance=parsed["guidance"],
            model=config.model,
            source="openai",
            fallback_used=False,
        )

    except Exception as exc:
        logger.exception("OpenAI classification failed: %s", exc)
        return _fallback_result(detections, f"OpenAI error: {exc.__class__.__name__}")


def classification_to_response_fields(result: ClassificationResult) -> dict[str, Any]:
    """Convert a classification result into API response fields."""
    return {
        "text": result.guidance,
        "items": result.items,
        "classifier": {
            "model": result.model,
            "source": result.source,
            "fallback_used": result.fallback_used,
        },
    }
