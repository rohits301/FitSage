"""The LLM path is tested against a fake client only; it has not been run against a real API."""
from types import SimpleNamespace

from app.services.advisor import RAGService
from app.services.llm import OpenAIAnswerer, validate


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.calls = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])


Q = "What are the side effects of taking a lot of magnesium?"


def service(text):
    client = FakeClient(text)
    return RAGService(answerer=OpenAIAnswerer(client, "fake-model")), client


def test_valid_citation_is_accepted():
    svc, client = service("High doses can cause diarrhea and cramping [mg-4].")
    out = svc.answer(Q)
    assert out.mode == "llm" and "[mg-4]" in out.answer
    prompt = client.calls[0]["messages"][1]["content"]
    assert "[mg-4]" in prompt and "Question:" in prompt


def test_citation_to_unretrieved_passage_falls_back_to_extractive():
    svc, _ = service("Magnesium cures everything [zn-3].")
    out = svc.answer(Q)
    assert out.mode == "extractive" and out.answer == out.sources[0].excerpt


def test_uncited_answer_falls_back():
    svc, _ = service("High doses can cause diarrhea.")
    assert svc.answer(Q).mode == "extractive"


def test_insufficient_evidence_reply_falls_back():
    svc, _ = service("INSUFFICIENT EVIDENCE")
    assert svc.answer(Q).mode == "extractive"


def test_validate_rejects_empty():
    assert validate("", []) is None
