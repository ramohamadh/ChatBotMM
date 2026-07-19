"""Pipeline wiring test: ask() routes to the generative engine over the real index.

Uses a stubbed GenerativeQA so no 3 GB model download is needed, but exercises
the real retriever + FAISS index built from data/docs.
"""

import pytest

from chatbot import config
from chatbot.rag import pipeline as pipeline_module
from chatbot.rag.pipeline import RAGPipeline


class _StubGen:
    def __init__(self, model_name=None, **kw):
        self.model_name = model_name

    def answer(self, question, chunks):
        return {
            "answer": f"پاسخ آزمایشی بر اساس {len(chunks)} منبع.",
            "score": 1.0,
            "source_chunks": [
                {"text": c.get("text", "")[:50], "metadata": c.get("metadata", {})}
                for c in chunks[:3]
            ],
        }


@pytest.fixture
def stub_generative(monkeypatch):
    monkeypatch.setattr(pipeline_module, "GenerativeQA", _StubGen)


index_exists = (config.VECTORSTORE_DIR / "faiss_index.index").exists()


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_ask_routes_to_generative_and_shapes_response(stub_generative):
    rag = RAGPipeline(
        docs_directory=str(config.DOCS_DIR),
        vectorstore_directory=str(config.VECTORSTORE_DIR),
        top_k=config.TOP_K,
        use_generative=True,
        # Must match the model the on-disk index was built with, or the
        # dimension check refuses to load it.
        embedding_model=config.EMBEDDING_MODEL,
        generative_model=config.GENERATIVE_MODEL,
        # Force the transformers backend so the stubbed GenerativeQA is used
        # (the llama.cpp backend would load a real GGUF model).
        generative_backend="transformers",
    )
    assert rag.use_generative is True
    assert type(rag.qa).__name__ == "_StubGen"
    assert rag.is_indexed is True

    resp = rag.ask("این سند درباره چیست؟")
    assert set(resp.keys()) >= {"answer", "score", "source_chunks"}
    assert resp["answer"]
    assert len(resp["source_chunks"]) <= 3
