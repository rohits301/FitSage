from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def ask(q):
    r = client.post("/api/ask", json={"question": q})
    assert r.status_code == 200
    return r.json()


def test_health_reports_corpus_size():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["generation"] == "extractive"
    assert body["passages"] >= 50 and body["sources"] >= 10


def test_vitamin_d_answer_cites_nih_and_quotes_the_page():
    body = ask("What is the daily vitamin D recommendation for grown-ups, in IU?")
    assert body["mode"] == "extractive"
    assert body["sources"][0]["id"] == "vitd-1"
    assert body["sources"][0]["url"].startswith("https://ods.od.nih.gov/")
    assert "15 to 20 mcg (600–800 IU) for adults" in body["answer"]


def test_pubmed_source_is_returned_for_protein_ceiling_question():
    body = ask("What did the meta-analysis conclude about protein supplements and strength gains?")
    assert any(s["organization"] == "PubMed" for s in body["sources"])


def test_uncovered_question_declines_without_sources():
    body = ask("Which running shoes should I buy?")
    assert body["mode"] == "declined" and body["reason"] == "no_evidence"
    assert body["sources"] == []


def test_urgent_question_is_escalated():
    body = ask("I have chest pain after a workout")
    assert body["mode"] == "declined" and body["reason"] == "urgent"
    assert body["sources"] == []


def test_personal_medical_question_is_declined():
    body = ask("Can you tell me if I have a vitamin deficiency based on my fatigue?")
    assert body["mode"] == "declined" and body["reason"] == "personal_medical"


def test_answer_text_is_always_a_verbatim_passage_in_extractive_mode():
    body = ask("Which foods contain magnesium?")
    assert body["answer"] == body["sources"][0]["excerpt"]


def test_input_validation():
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
