import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes

load_dotenv()

app = FastAPI(
    title="Promtior Chatbot",
    description="RAG-powered chatbot for Promtior AI",
    version="1.0.0",
)

from chain import rag_chain  # noqa: E402

add_routes(app, rag_chain, path="/chat")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY is not set. "
            "Vectorstore will not be built."
        )
        return
    try:
        from app.ingest import get_or_create_vectorstore
        logger.info("Building vectorstore on startup...")
        get_or_create_vectorstore()
        logger.info("Vectorstore ready.")
    except Exception as e:
        logger.error(f"Vectorstore build failed: {e}")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)