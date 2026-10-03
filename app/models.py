from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    id: str
    organization: str
    title: str
    url: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
    safety_note: str

