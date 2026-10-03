from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import AskRequest, AskResponse
from app.services.advisor import RAGService
from app.services.llm import OpenAIAnswerer
from app.services.retrieval import load_corpus

ROOT = Path(__file__).parent
app = FastAPI(title="FitSage", version="0.2.0", description="Citation-first nutrition and exercise retrieval")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def build_service() -> RAGService:
    settings = get_settings()
    answerer = None
    if settings.generation.lower() == "openai" and settings.llm_configured:
        answerer = OpenAIAnswerer.from_settings(settings)
    return RAGService(answerer=answerer)


service = build_service()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    corpus = load_corpus()
    return {
        "status": "ok",
        "generation": "llm" if service.answerer else "extractive",
        "passages": len(corpus),
        "sources": len({p.url for p in corpus}),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    return service.answer(payload.question)
