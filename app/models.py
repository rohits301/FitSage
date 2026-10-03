from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    id: str
    organization: str
    title: str
    section: str
    kind: str
    url: str
    excerpt: str
    confidence: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str  # extractive | llm | declined
    reason: Optional[str] = None  # why the system declined, when it did
    safety_note: str
