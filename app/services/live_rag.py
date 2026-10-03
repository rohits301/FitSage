"""Production integration boundary for OpenAI, LangChain, and Pinecone.

This module is intentionally not activated until credentials and an approved,
versioned source index exist. Keeping it separate prevents a demo from silently
calling paid services or presenting uncited model output as evidence.
"""

from app.config import Settings


class LiveRAGService:
    def __init__(self, settings: Settings) -> None:
        if not settings.live_configured:
            raise RuntimeError("Live mode requires OpenAI and Pinecone configuration.")
        self.settings = settings

    def answer(self, question: str):  # pragma: no cover - integration work
        raise NotImplementedError(
            "Complete the README production checklist before enabling live mode."
        )

