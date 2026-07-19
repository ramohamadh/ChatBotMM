"""
REST API for ChatBotMM, built with FastAPI.

A thin HTTP shell over the same RAGPipeline the CLI uses:

    POST /ask     {"question": "..."}  -> answer + score + sources
    POST /index   {"force": false}     -> (re)index the documents in data/docs
    GET  /stats                        -> index statistics
    GET  /health                       -> liveness / readiness info

Run it with:

    chatbot serve                # or:
    uvicorn chatbot.api:app --host 0.0.0.0 --port 8000

The answer model is loaded once at startup (lifespan) so the first request
doesn't pay the model-loading cost. Generation is CPU-bound and the llama.cpp
backend is not thread-safe, so questions are answered one at a time behind a
lock; concurrent requests queue up in FastAPI's threadpool.
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ask_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    return_context: bool = Field(
        False, description="Include the retrieved chunks in the response."
    )


class SourceChunk(BaseModel):
    text: str
    metadata: dict = {}
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    score: float
    sources: list[str] = []
    timings: dict[str, float] = {}
    retrieved_chunks: list[SourceChunk] | None = None


class IndexRequest(BaseModel):
    force: bool = Field(False, description="Rebuild the index even if one exists.")


class IndexResponse(BaseModel):
    total_chunks: int = 0
    total_documents: int = 0


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    total_chunks: int = 0


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load the pipeline (and the answer model) once, before serving."""
    from .commands import get_default_rag_pipeline

    logger.info("Loading RAG pipeline...")
    pipeline = get_default_rag_pipeline()
    if pipeline.is_indexed:
        # Load the heavy answer model now so the first /ask is fast.
        pipeline.warm_up()
    app.state.pipeline = pipeline
    logger.info("RAG pipeline ready")
    yield


app = FastAPI(
    title="ChatBotMM API",
    description="Fully local Persian/English RAG document question-answering.",
    version="0.2.0",
    lifespan=_lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = app.state.pipeline
    stats = pipeline.get_stats()
    return HealthResponse(
        status="ok",
        indexed=pipeline.is_indexed,
        total_chunks=stats.get("total_chunks", 0),
    )


@app.get("/stats")
def stats() -> dict:
    return app.state.pipeline.get_stats()


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    """(Re)index the documents in the docs directory."""
    pipeline = app.state.pipeline
    with _ask_lock:  # don't reindex while a question is being answered
        try:
            result = pipeline.index_documents(force_reindex=request.force)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
    if pipeline.is_indexed:
        pipeline.warm_up()
    return IndexResponse(
        total_chunks=result.get("total_chunks", 0),
        total_documents=result.get("total_documents", 0),
    )


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents."""
    pipeline = app.state.pipeline
    if not pipeline.is_indexed:
        raise HTTPException(
            status_code=409,
            detail="No index found. POST /index first (documents go in data/docs).",
        )

    with _ask_lock:
        response = pipeline.ask(request.question, return_context=request.return_context)

    sources = sorted(
        {
            chunk.get("metadata", {}).get("filename", "")
            for chunk in response.get("source_chunks", [])
            if isinstance(chunk, dict)
        }
        - {""}
    )
    return AskResponse(
        answer=response["answer"],
        score=response["score"],
        sources=sources,
        timings=response.get("timings", {}),
        retrieved_chunks=(
            [
                SourceChunk(
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    score=chunk.get("score"),
                )
                for chunk in response["retrieved_chunks"]
            ]
            if request.return_context
            else None
        ),
    )
