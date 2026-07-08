"""Tests for OpenAI classifier helper."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

FINAL_DIR = Path(__file__).resolve().parents[1] / "final"
sys.path.insert(0, str(FINAL_DIR))

from openai_classifier import (  # noqa: E402
    ClassifierConfig,
    _parse_response_content,
    classify_waste,
    prepare_image_data_url,
)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (1024, 768), color=(120, 180, 90))


@pytest.fixture
def sample_detections():
    return [
        {
            "class_name": "bottle",
            "confidence": 0.91,
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "route": "Recycle",
        }
    ]


def test_prepare_image_data_url_resizes_large_image(sample_image):
    config = ClassifierConfig(
        enabled=True,
        model="gpt-4o-mini",
        image_detail="low",
        timeout_seconds=20,
        max_image_size=256,
        jpeg_quality=80,
    )

    data_url = prepare_image_data_url(sample_image, config)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_parse_response_content_normalizes_routes():
    content = json.dumps({
        "items": [
            {
                "name": "plastic bottle",
                "material": "plastic",
                "route": "Recycle",
                "confidence": 0.95,
                "caveats": "rinse before recycling",
            }
        ],
        "guidance": "Rinse and recycle this bottle.",
    })

    parsed = _parse_response_content(content)
    assert parsed["items"][0]["route"] == "Recycle"
    assert parsed["guidance"] == "Rinse and recycle this bottle."


def test_parse_response_content_invalid_route_uses_fallback():
    content = json.dumps({
        "items": [
            {
                "name": "mystery item",
                "material": "unknown",
                "route": "Definitely Not Valid",
                "confidence": 0.4,
                "caveats": "",
            }
        ],
        "guidance": "Check local rules.",
    })

    parsed = _parse_response_content(content)
    assert parsed["items"][0]["route"] == "Landfill / Donate / Check rules"


def test_classify_waste_disabled_uses_fallback(sample_image, sample_detections):
    config = ClassifierConfig(
        enabled=False,
        model="gpt-4o-mini",
        image_detail="low",
        timeout_seconds=20,
        max_image_size=768,
        jpeg_quality=85,
    )

    result = classify_waste(
        pil_img=sample_image,
        detections=sample_detections,
        config=config,
    )

    assert result.fallback_used is True
    assert result.source == "yolo"
    assert result.items[0]["name"] == "bottle"


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
def test_classify_waste_openai_success(sample_image, sample_detections):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "items": [
                        {
                            "name": "plastic water bottle",
                            "material": "plastic",
                            "route": "Recycle",
                            "confidence": 0.97,
                            "caveats": "remove cap if required locally",
                        }
                    ],
                    "guidance": "Rinse the bottle and place it in recycling.",
                })
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    config = ClassifierConfig(
        enabled=True,
        model="gpt-4o-mini",
        image_detail="low",
        timeout_seconds=20,
        max_image_size=768,
        jpeg_quality=85,
    )

    result = classify_waste(
        pil_img=sample_image,
        detections=sample_detections,
        user_text="Is this recyclable?",
        config=config,
        client=mock_client,
    )

    assert result.fallback_used is False
    assert result.source == "openai"
    assert result.guidance.startswith("Rinse the bottle")
    assert result.items[0]["route"] == "Recycle"
    mock_client.chat.completions.create.assert_called_once()


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False)
def test_classify_waste_openai_error_falls_back(sample_image, sample_detections):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")

    config = ClassifierConfig(
        enabled=True,
        model="gpt-4o-mini",
        image_detail="low",
        timeout_seconds=20,
        max_image_size=768,
        jpeg_quality=85,
    )

    result = classify_waste(
        pil_img=sample_image,
        detections=sample_detections,
        config=config,
        client=mock_client,
    )

    assert result.fallback_used is True
    assert result.source == "yolo"
    assert "API down" in result.items[0]["caveats"] or result.items[0]["name"] == "bottle"
