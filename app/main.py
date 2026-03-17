import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes

load_dotenv()

logger = logging.getLogger("uvicorn")

_chain = None
_chain_lock = threading.Lock()


def get_chain():
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                from .ingest import get_or_create_vectorstore
                from .chain import build_rag_chain
                logger.info("Building vectorstore...")
                vectorstore = get_or_create_vectorstore()
                _chain = build_rag_chain(vectorstore)
                logger.info("RAG chain ready.")
    return _chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
    else:
        try:
            chain = get_chain()
            add_routes(app, chain, path="/chat")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
    yield


app = FastAPI(
    title="Promtior Chatbot",
    description="RAG-powered chatbot for Promtior AI",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
