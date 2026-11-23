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


@pytest.mark.parametrize(
    ("with_photo", "note", "basis"),
    [(True, "", "image"), (False, "Used AA battery", "note"), (True, "It's an AA", "image_and_note")],
)
def test_predict_adaption_uses_submitted_inputs(client, monkeypatch, with_photo, note, basis):
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_ENABLED", "true")
    monkeypatch.delenv("ADAPTION_MODEL", raising=False)
    monkeypatch.setattr(flask_app, "USE_YOLO", False)
    result_text = json.dumps({
        "items": [{"name": "AA battery", "route": "Hazardous Waste", "bin": "Blue Bin (Recycling)"}],
        "guidance": "Keep the battery out of city carts.",
    })
    data = {"text": note}
    if with_photo:
        data["file"] = (_make_image_bytes(), "item.png")
    with patch("openai_classifier.httpx.post") as post:
        post.return_value.json.return_value = {
            "status": "completed",
            "output": [{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": result_text}],
            }],
        }
        response = client.post("/predict", data=data, content_type="multipart/form-data", headers=AUTH_HEADERS)

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["classifier"] == {
        "model": "gemma-4-26b-a4b-it", "source": "adaption", "basis": basis, "fallback_used": False,
    }
    assert payload["items"][0]["bin"] == "Special Drop-off"
    assert payload["text"] == "Keep the battery out of city carts."
    post.assert_called_once()
    assert post.call_args.args[0] == "https://api.prod.adaptionlabs.ai/api/v1/responses"
    options = post.call_args.kwargs
    assert options["headers"]["Authorization"] == "Bearer test-adaption-key"
    assert options["json"]["store"] is False
    parts = options["json"]["input"][0]["content"]
    assert parts[0]["type"] == "input_text"
    assert note in parts[0]["text"]
    if with_photo:
        assert parts[1]["type"] == "input_image"
        assert parts[1]["image_url"].startswith("data:image/jpeg;base64,")
    else:
        assert len(parts) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"status": "incomplete", "output_text": "{}"},
        {"status": "completed", "output_text": "not JSON"},
        {"output_text": '{"items": {}, "guidance": "bogus"}'},
        {"output_text": '{"items": [{"name":"jar","confidence":NaN}],"guidance":"test"}'},
        {"output": []},
    ],
)
def test_unusable_adaption_response_has_no_fabricated_item(client, monkeypatch, body):
    monkeypatch.setenv("ADAPTION_API_KEY", "test-adaption-key")
    monkeypatch.setenv("ADAPTION_ENABLED", "true")
    with patch("openai_classifier.httpx.post") as post:
        post.return_value.json.return_value = body
        response = client.post("/predict", data={"text": "mystery item"}, headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["items"] == []
    assert payload["classifier"]["fallback_used"]
    assert payload["classifier"]["basis"] == "none"
    post.assert_called_once()


@patch("app.classify_waste")
def test_invalid_inputs_do_not_reach_provider(classify, client):
    requests = [
        {"file": (io.BytesIO(b"not an image"), "image.jpg")},
        {"text": "   "},
        {"text": "x" * 1001},
    ]
    for data in requests:
        response = client.post("/predict", data=data, headers=AUTH_HEADERS)
        assert response.status_code == 400
        assert response.get_json()["error"]
    classify.assert_not_called()


@patch("app.classify_waste")
def test_predict_rejects_oversized_photo_before_classification(classify, client, monkeypatch):
    monkeypatch.setattr(flask_app, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/predict", data={"file": (_make_image_bytes(), "test.png")}, headers=AUTH_HEADERS)
    assert response.status_code == 413
    assert response.get_json()["error"]
    classify.assert_not_called()


def test_predict_redacts_provider_failure(client, monkeypatch, caplog):
    monkeypatch.setenv("ADAPTION_API_KEY", "fake-key")
    monkeypatch.setenv("ADAPTION_ENABLED", "true")
    with patch("openai_classifier.httpx.post", side_effect=TimeoutError("private-request-content")) as post:
        response = client.post("/predict", data={"text": "item description"}, headers=AUTH_HEADERS)
    assert response.get_json()["classifier"]["fallback_used"]
    assert "private-request-content" not in caplog.text
    assert "private-request-content" not in response.get_data(as_text=True)
    post.assert_called_once()


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
