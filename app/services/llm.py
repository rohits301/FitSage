"""Optional LLM answer writer, constrained to retrieved passages.

NOT used by default and NOT part of the reported evaluation. It is exercised in the
test-suite only against a fake client. Run it once against the real API with your own
key before describing it as working.

Guard rails, all enforced in code rather than by trusting the model:
  * the prompt contains only the retrieved passages, each tagged with its id;
  * the model must cite passage ids in square brackets;
  * if it cites an id that was not retrieved, cites nothing, or returns nothing,
    the caller falls back to the extractive answer.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from app.services.retrieval import Hit

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[([a-z0-9_]+-\d+)\]")

SYSTEM_PROMPT = (
    "You answer nutrition and exercise questions using ONLY the numbered evidence passages "
    "provided. Write at most three sentences. After each claim, cite the passage id in square "
    "brackets, e.g. [mg-4]. If the passages do not answer the question, reply exactly: "
    "INSUFFICIENT EVIDENCE. Do not give personal medical advice or add facts that are not in the passages."
)


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict]:
    evidence = "\n\n".join(f"[{h.passage.id}] ({h.passage.title}) {h.passage.text}" for h in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence passages:\n{evidence}\n\nQuestion: {question}"},
    ]


def validate(text: str, hits: Sequence[Hit]) -> Optional[str]:
    """Return the answer if every citation refers to a retrieved passage, else None."""
    text = (text or "").strip()
    if not text or "INSUFFICIENT EVIDENCE" in text.upper():
        return None
    cited = _CITATION.findall(text)
    allowed = {h.passage.id for h in hits}
    if not cited or any(c not in allowed for c in cited):
        return None
    return text


class OpenAIAnswerer:
    """Thin wrapper over an OpenAI-compatible chat-completions client."""

    def __init__(self, client, model: str) -> None:
        self.client, self.model = client, model

    @classmethod
    def from_settings(cls, settings) -> "OpenAIAnswerer":
        from openai import OpenAI  # imported lazily: only needed in this mode

        return cls(OpenAI(api_key=settings.openai_api_key, timeout=20.0, max_retries=1), settings.openai_model)

    def answer(self, question: str, hits: Sequence[Hit]) -> Optional[str]:
        """Return a validated, cited answer, or None for any failure (the caller then
        falls back to the verbatim passage). API errors, timeouts and malformed responses
        are all treated as "no answer written"."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=build_messages(question, hits), temperature=0
            )
            return validate(resp.choices[0].message.content, hits)
        except Exception as exc:  # noqa: BLE001 - deliberate: never let the optional writer break an answer
            log.warning("LLM answer writer failed (%s); using extractive answer", type(exc).__name__)
            return None
