"""Integration tests for Flask /predict endpoint."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

FINAL_DIR = Path(__file__).resolve().parents[1] / "final"
sys.path.insert(0, str(FINAL_DIR))

import app as flask_app  # noqa: E402
from openai_classifier import ClassificationResult  # noqa: E402

TEST_API_KEY = "test-secret-key-123"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(flask_app, "BACKEND_API_KEY", TEST_API_KEY)
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as test_client:
        yield test_client


def _make_image_bytes():
    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _mock_yolo_result():
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.88]
    mock_box.xyxy = [[1.0, 2.0, 30.0, 40.0]]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "bottle"}

    return [mock_result]


@patch("app.classify_waste")
@patch("app.model")
def test_predict_success_with_openai_fields(mock_model, mock_classify, client, monkeypatch):
    monkeypatch.setattr(flask_app, "USE_YOLO", True)
    monkeypatch.setattr(flask_app, "COCO_TO_BIN", {"bottle": "Recycle"})
    mock_model.predict.return_value = _mock_yolo_result()
    mock_classify.return_value = ClassificationResult(
        items=[
            {
                "name": "plastic bottle",
                "material": "plastic",
                "route": "Recycle",
                "confidence": 0.95,
                "caveats": "",
            }
        ],
        guidance="Rinse and recycle this bottle.",
        model="gpt-4o-mini",
        source="openai",
        fallback_used=False,
    )

    response = client.post(
        "/predict",
        data={
            "file": (_make_image_bytes(), "test.png"),
            "text": "Can I recycle this?",
        },
        content_type="multipart/form-data",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["objects"] == ["bottle"]
    assert payload["detections"][0]["class_name"] == "bottle"
    assert payload["bin_totals"]["Recycle"] == 1
    assert payload["text"] == "Rinse and recycle this bottle."
    assert payload["classifier"]["source"] == "openai"
    assert payload["classifier"]["fallback_used"] is False
    assert len(payload["items"]) == 1

    mock_classify.assert_called_once()
    _, kwargs = mock_classify.call_args
    assert kwargs["user_text"] == "Can I recycle this?"


@patch("app.classify_waste")
@patch("app.model")
def test_predict_fallback_preserves_legacy_fields(mock_model, mock_classify, client, monkeypatch):
    monkeypatch.setattr(flask_app, "USE_YOLO", True)
    mock_model.predict.return_value = _mock_yolo_result()
    mock_classify.return_value = ClassificationResult(
        items=[
            {
                "name": "bottle",
                "material": "unknown",
                "route": "Recycle",
                "confidence": 0.88,
                "caveats": "OpenAI classifier disabled",
            }
        ],
        guidance="Classification used detection-only fallback. Suggested routes: Recycle.",
        model="fallback",
        source="yolo",
        fallback_used=True,
    )

    response = client.post(
        "/predict",
        data={"file": (_make_image_bytes(), "test.png")},
        content_type="multipart/form-data",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert "objects" in payload
    assert "detections" in payload
    assert "bin_totals" in payload
    assert payload["text"]
    assert payload["classifier"]["fallback_used"] is True


@patch("app.classify_waste")
def test_predict_skips_yolo_when_disabled(mock_classify, client, monkeypatch):
    monkeypatch.setattr(flask_app, "USE_YOLO", False)
    mock_classify.return_value = ClassificationResult(
        items=[
            {
                "name": "plastic bottle",
                "material": "plastic",
                "route": "Recycle",
                "confidence": 0.95,
                "caveats": "",
            }
        ],
        guidance="Rinse and recycle this bottle.",
        model="gpt-4o-mini",
        source="openai",
        fallback_used=False,
    )

    response = client.post(
        "/predict",
        data={
            "file": (_make_image_bytes(), "test.png"),
            "text": "Can I recycle this?",
        },
        content_type="multipart/form-data",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["objects"] == []
    assert payload["detections"] == []
    assert payload["bin_totals"] == {}
    assert payload["classifier"]["source"] == "openai"

    mock_classify.assert_called_once()
    _, kwargs = mock_classify.call_args
    assert kwargs["detections"] == []
    assert kwargs["user_text"] == "Can I recycle this?"


def test_predict_missing_file_returns_400(client):
    response = client.post("/predict", data={}, headers=AUTH_HEADERS)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_missing_api_key(client):
    response = client.post(
        "/predict",
        data={"file": (_make_image_bytes(), "test.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 401
    assert "error" in response.get_json()


def test_predict_rejects_wrong_api_key(client):
    response = client.post(
        "/predict",
        data={"file": (_make_image_bytes(), "test.png")},
        content_type="multipart/form-data",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
    assert "error" in response.get_json()


def test_predict_rejects_malformed_auth_header(client):
    response = client.post(
        "/predict",
        data={"file": (_make_image_bytes(), "test.png")},
        content_type="multipart/form-data",
        headers={"Authorization": TEST_API_KEY},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "origin",
    [
        "https://scrapp.app",
        "https://www.scrapp.app",
        "https://scrapp-sd.vercel.app",
    ],
)
def test_cors_allows_configured_origins(client, origin):
    response = client.options(
        "/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("Access-Control-Allow-Origin") == origin


def test_cors_rejects_unlisted_origin(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("Access-Control-Allow-Origin") is None
