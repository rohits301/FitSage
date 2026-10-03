from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import AskRequest, AskResponse
from app.services.advisor import DemoRAGService

ROOT = Path(__file__).parent
app = FastAPI(title="FitSage", version="0.1.0", description="Citation-first fitness RAG demo")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
service = DemoRAGService()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "mode": settings.app_mode, "live_configured": str(settings.live_configured).lower()}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    settings = get_settings()
    if settings.app_mode.lower() == "live":
        raise HTTPException(
            status_code=501,
            detail="Live mode is intentionally pending the documented source-index integration.",
        )
    return service.answer(payload.question)

