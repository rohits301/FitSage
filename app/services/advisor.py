from app.models import AskResponse, Source
from app.services.retrieval import DemoRetriever
from typing import Optional

SAFETY_NOTE = (
    "Educational information only—not medical advice. For symptoms, pregnancy, "
    "conditions, medications, or supplement decisions, consult a qualified clinician."
)

_URGENT_TERMS = {"chest pain", "fainting", "overdose", "suicidal", "emergency"}


class DemoRAGService:
    def __init__(self, retriever: Optional[DemoRetriever] = None) -> None:
        self.retriever = retriever or DemoRetriever()

    def answer(self, question: str) -> AskResponse:
        normalized = question.lower()
        if any(term in normalized for term in _URGENT_TERMS):
            return AskResponse(
                answer=(
                    "This may need urgent, in-person help. Contact local emergency "
                    "services or a qualified health professional now rather than relying "
                    "on an online assistant."
                ),
                sources=[],
                mode="demo",
                safety_note=SAFETY_NOTE,
            )

        sources = self.retriever.retrieve(question)
        if not sources:
            return AskResponse(
                answer=(
                    "I don’t have enough evidence in the current curated demo corpus "
                    "to answer that responsibly. Try a question about vitamin D, "
                    "magnesium, balanced meals, protein, or resistance training."
                ),
                sources=[],
                mode="demo",
                safety_note=SAFETY_NOTE,
            )

        lead = sources[0]
        answer = (
            f"Based on the retrieved {lead.organization} material: {lead.excerpt} "
            "This is a research summary, so it should not be treated as a personalized "
            "diet or training prescription."
        )
        return AskResponse(answer=answer, sources=sources, mode="demo", safety_note=SAFETY_NOTE)
