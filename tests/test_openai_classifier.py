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
    SYSTEM_PROMPT,
    _build_user_content,
    _fallback_result,
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
    assert parsed["items"][0]["bin"] == "Blue Bin (Recycling)"
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
    assert parsed["items"][0]["bin"] == "Special Drop-off"


def test_parse_response_content_orders_waste_items_before_person():
    content = json.dumps({
        "items": [
            {
                "name": "person",
                "material": "unknown",
                "route": "Living Things",
                "confidence": 0.99,
                "caveats": "",
            },
            {
                "name": "USB cable",
                "material": "plastic and metal",
                "route": "E-Waste",
                "confidence": 0.82,
                "caveats": "remove from charger if attached",
            },
        ],
        "guidance": "Recycle this cable as e-waste.",
    })

    parsed = _parse_response_content(content)
    assert parsed["items"][0]["name"] == "USB cable"
    assert parsed["items"][0]["route"] == "E-Waste"


def test_build_user_content_filters_non_waste_yolo_hints():
    detections = [
        {
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "route": "Living Things",
        },
        {
            "class_name": "mouse",
            "confidence": 0.4,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "route": "E-Waste",
        },
    ]
    config = ClassifierConfig(
        enabled=True,
        model="gpt-4o-mini",
        image_detail="high",
        timeout_seconds=20,
        max_image_size=1280,
        jpeg_quality=85,
    )

    content = _build_user_content(
        image_data_url="data:image/jpeg;base64,abc",
        detections=detections,
        user_text=None,
        config=config,
    )

    payload_text = content[0]["text"]
    assert "person" not in payload_text
    assert "mouse" in payload_text
    assert content[1]["image_url"]["detail"] == "high"


def test_system_prompt_mentions_e_waste_accessories():
    assert "USB cables" in SYSTEM_PROMPT
    assert "held or foreground waste item" in SYSTEM_PROMPT


def test_system_prompt_contains_san_diego_rules():
    assert "City of San Diego" in SYSTEM_PROMPT
    assert "Blue Bin (Recycling)" in SYSTEM_PROMPT
    assert "Green Bin (Organics)" in SYSTEM_PROMPT
    assert "Gray Bin (Trash)" in SYSTEM_PROMPT
    # SD quirks per the current What Goes Where guide: clean foam accepted in the
    # blue bin, plastic bags/film never, batteries to Miramar HHW.
    assert "Styrofoam packaging goes in the blue bin" in SYSTEM_PROMPT
    assert "NEVER in the blue bin: packing peanuts" in SYSTEM_PROMPT
    assert "plastic bags" in SYSTEM_PROMPT.lower()
    assert "Miramar" in SYSTEM_PROMPT
    assert "858-694-7000" in SYSTEM_PROMPT


def test_fallback_result_ignores_person_only_detections():
    detections = [
        {
            "class_name": "person",
            "confidence": 0.95,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "route": "Living Things",
        }
    ]

    result = _fallback_result(detections, "OpenAI classifier disabled")

    assert result.items == []
    assert "No items were confidently detected" in result.guidance


def test_fallback_result_assigns_route_default_bin(sample_detections):
    result = _fallback_result(sample_detections, "OpenAI classifier disabled")

    assert result.items[0]["route"] == "Recycle"
    assert result.items[0]["bin"] == "Blue Bin (Recycling)"


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
                            "bin": "Blue Bin (Recycling)",
                            "confidence": 0.97,
                            "caveats": "empty and dry before the blue bin",
                        }
                    ],
                    "guidance": "Rinse the bottle and place it in your blue bin.",
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
    assert result.items[0]["bin"] == "Blue Bin (Recycling)"
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
