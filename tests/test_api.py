from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_rejects_empty_question():
    res = client.post("/chat", json={"question": ""})
    assert res.status_code == 400


def test_home_page_loads():
    res = client.get("/")
    assert res.status_code == 200
    assert "RAG-Chatbot" in res.text
