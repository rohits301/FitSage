import re
from typing import Optional

from app.models import AskResponse, Source
from app.services.llm import OpenAIAnswerer
from app.services.retrieval import Hit, HybridRetriever, load_corpus

SAFETY_NOTE = (
    "Educational information only—not medical advice. For symptoms, pregnancy, "
    "conditions, medications, or supplement decisions, consult a qualified clinician."
)

# Rule-based guards. These are simple phrase matches, not a classifier, and they are
# covered by unit tests only -- the retrieval evaluation does not measure them.
_URGENT = re.compile(r"chest pain|faint|overdose|suicid|can'?t breathe|emergency|poison", re.I)
_PERSONAL_MEDICAL = re.compile(
    r"\bdo i have\b|\bam i (?:sick|deficient|anemic)\b|\bdiagnos|what'?s wrong with me"
    r"|\bshould i stop taking\b|\bchange my (?:dose|medication)\b|\bif i have\b",
    re.I,
)


class AnswerService:
    """Hybrid retrieval, then a cited answer, or an explicit refusal.

    The default answer is the top retrieved passage, verbatim. An LLM writer can be
    plugged in (``answerer``); if it is absent or its output fails citation checks the
    extractive answer is used.
    """

    def __init__(self, retriever: Optional[HybridRetriever] = None, answerer: Optional[OpenAIAnswerer] = None) -> None:
        self.retriever = retriever or HybridRetriever.from_config(load_corpus())
        self.answerer = answerer

    @staticmethod
    def _declined(answer: str, reason: str) -> AskResponse:
        return AskResponse(answer=answer, sources=[], mode="declined", reason=reason, safety_note=SAFETY_NOTE)

    @staticmethod
    def _source(h: Hit) -> Source:
        p = h.passage
        return Source(
            id=p.id, organization=p.organization, title=p.title, section=p.section,
            kind=p.kind, url=p.url, excerpt=p.text, confidence=round(h.confidence, 3),
        )

    def answer(self, question: str) -> AskResponse:
        if _URGENT.search(question):
            return self._declined(
                "This may need urgent, in-person help. Contact local emergency services or a "
                "qualified health professional now rather than relying on an online assistant.",
                "urgent",
            )
        if _PERSONAL_MEDICAL.search(question):
            return self._declined(
                "I can share general, sourced information, but I can't assess your own symptoms, "
                "diagnose a deficiency, or advise on changing a medicine. Please ask a clinician or pharmacist.",
                "personal_medical",
            )

        hits = self.retriever.search(question, k=3)
        if not hits:
            return self._declined(
                "I couldn't find evidence for that in the current corpus, so I won't guess. "
                "It covers selected NIH, MedlinePlus and PubMed material on vitamins, minerals, "
                "supplements for exercise, and resistance training.",
                "no_evidence",
            )

        sources = [self._source(h) for h in hits]
        if self.answerer is not None:
            written = self.answerer.answer(question, hits)
            if written:
                return AskResponse(answer=written, sources=sources, mode="llm", safety_note=SAFETY_NOTE)
        lead = hits[0].passage
        return AskResponse(answer=lead.text, sources=sources, mode="extractive", safety_note=SAFETY_NOTE)
