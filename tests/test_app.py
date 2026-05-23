import sys; sys.path.insert(0, "src")
from unittest.mock import patch
from fastapi.testclient import TestClient

# Patch startup warmup so no real model loads during import
with patch("rag._ensure_loaded", return_value=(None, None, None)):
    from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_ask_validates_empty_query():
    r = client.post("/ask", json={"query": "", "backend": "aistudio", "k": 3})
    assert r.status_code == 422  # pydantic rejects min_length

def test_ask_validates_bad_backend():
    r = client.post("/ask", json={"query": "hi", "backend": "openai", "k": 3})
    assert r.status_code == 422  # pattern rejects unknown backend

def test_ask_returns_answer():
    fake = {"query": "hi", "answer": "A", "backend": "aistudio",
            "sources": [], "tokens": {"in": 1, "out": 1}}
    with patch("app.answer", return_value=fake):
        r = client.post("/ask", json={"query": "hi", "backend": "aistudio", "k": 3})
    assert r.status_code == 200
    assert r.json()["answer"] == "A"
