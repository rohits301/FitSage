from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_demo_mode():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "demo"


def test_vitamin_d_answer_has_nih_citation():
    response = client.post("/api/ask", json={"question": "How much vitamin D do adults need?"})
    body = response.json()
    assert response.status_code == 200
    assert body["sources"][0]["id"] == "nih-vitamin-d"
    assert "600 IU" in body["answer"]


def test_uncovered_question_declines_cleanly():
    response = client.post("/api/ask", json={"question": "Which running shoes should I buy?"})
    body = response.json()
    assert response.status_code == 200
    assert body["sources"] == []
    assert "don’t have enough evidence" in body["answer"]


def test_urgent_question_is_escalated():
    response = client.post("/api/ask", json={"question": "I have chest pain after a workout"})
    body = response.json()
    assert response.status_code == 200
    assert body["sources"] == []
    assert "urgent" in body["answer"].lower()
