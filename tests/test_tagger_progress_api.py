"""Tagger progress API smoke tests (no ONNX / no real images)."""

from fastapi.testclient import TestClient

from mikazuki.app.application import app
from mikazuki.tagger.model_fetch import use_download_endpoint
from mikazuki.tagger.progress import tagger_progress


def test_tagger_status_idle():
    tagger_progress.reset_idle("test")
    client = TestClient(app)
    r = client.get("/api/tagger/status")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["data"]["phase"] == "idle"


def test_tagger_busy_guard():
    tagger_progress.reset_idle("test")
    assert tagger_progress.try_begin("downloading", "wd14-convnextv2-v2", "busy test")
    client = TestClient(app)
    r = client.post("/api/tagger/prefetch", json={"interrogator_model": "wd14-convnextv2-v2"})
    assert r.json()["status"] == "fail"
    tagger_progress.release()
    tagger_progress.reset_idle("test")


def test_tagger_html_serves_progress_script():
    client = TestClient(app)
    r = client.get("/tagger.html")
    assert r.status_code == 200
    assert "tagger-progress.js" in r.text


def test_tagger_default_download_endpoint_preserves_existing_hf_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

    with use_download_endpoint(""):
        import os

        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
