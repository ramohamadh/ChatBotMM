"""Tests for reranker wiring and neighbor stitching (no models loaded)."""

import pytest

from chatbot import config
from chatbot.rag import pipeline as pipeline_module
from chatbot.rag.pipeline import RAGPipeline


class _StubGen:
    def __init__(self, model_name=None, **kw):
        pass

    def answer(self, question, chunks):
        return {
            "answer": f"پاسخ بر اساس {len(chunks)} قطعه.",
            "score": 1.0,
            "source_chunks": [],
        }


class _StubReranker:
    """Reverses candidate order so tests can tell it actually ran."""

    def __init__(self):
        self.calls = []

    def rerank(self, question, candidates, top_k):
        self.calls.append((question, len(candidates), top_k))
        reversed_chunks = [chunk for chunk, _ in reversed(candidates)]
        return [(chunk, 0.9) for chunk in reversed_chunks[:top_k]]


@pytest.fixture
def stub_generative(monkeypatch):
    monkeypatch.setattr(pipeline_module, "GenerativeQA", _StubGen)


index_exists = (config.VECTORSTORE_DIR / "faiss_index.index").exists()


def _pipeline(**overrides):
    kwargs = dict(
        docs_directory=str(config.DOCS_DIR),
        vectorstore_directory=str(config.VECTORSTORE_DIR),
        embedding_model=config.EMBEDDING_MODEL,
        use_generative=True,
        generative_backend="transformers",
        rerank_enabled=False,  # never load the real cross-encoder in tests
    )
    kwargs.update(overrides)
    return RAGPipeline(**kwargs)


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_reranker_is_used_and_narrows_candidates(stub_generative):
    rag = _pipeline(rerank_candidates=10, rerank_top_k=3)
    stub = _StubReranker()
    rag.rerank_enabled = True
    rag._reranker = stub

    # Deliberately NOT an overview question ("... سند چیست") — that path
    # prepends intro chunks with score 1.0, which would mask the stub's score.
    resp = rag.ask("شناسه یکتای حافظه مالیاتی چگونه ساخته می‌شود؟")
    assert resp["answer"]
    assert len(stub.calls) == 1
    _, n_candidates, top_k = stub.calls[0]
    assert n_candidates == 10
    assert top_k == 3
    # Confidence comes from the cross-encoder probability (0.9 -> ~90%).
    assert resp["confidence"] == round(0.9 * 94 + 5)


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_rare_token_chunk_survives_reranker(stub_generative):
    """If the cross-encoder drops every rare-token chunk, one is re-inserted."""
    rag = _pipeline(rerank_top_k=3)
    chunks = rag.vectorstore.chunks
    if len(chunks) < 5:
        pytest.skip("index too small")
    rare_chunk = chunks[4]
    others = [(c, 0.5) for c in chunks[:4]]

    class _DroppingReranker:
        def rerank(self, question, candidates, top_k):
            # Pretend the cross-encoder ranked the rare chunk dead last.
            kept = [(c, 0.8) for c, _ in candidates if c is not rare_chunk]
            return kept[:top_k]

    def fake_retrieve(question, top_k=5):
        rag.retriever.last_rare_chunk_ids = {id(rare_chunk)}
        return [(rare_chunk, 0.9)] + others

    rag.rerank_enabled = True
    rag._reranker = _DroppingReranker()
    rag.retriever.retrieve = fake_retrieve

    resp = rag.ask("کد فیلد tinb چیست؟", return_context=True)
    texts = [c["text"] for c in resp["retrieved_chunks"]]
    assert rare_chunk["text"] in texts
    # The reranker's own top pick still leads; the rare chunk comes second.
    assert texts.index(rare_chunk["text"]) >= 1


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_overview_question_has_no_confidence(stub_generative):
    """Intro chunks carry a hand-assigned 1.0 score — no fake confidence."""
    rag = _pipeline()
    resp = rag.ask("این سند در مورد چیست؟")
    assert resp["confidence"] is None


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_disabled_reranker_never_loads(stub_generative):
    rag = _pipeline()
    assert rag.reranker is None
    resp = rag.ask("شناسه سند چیست؟")
    assert resp["answer"]


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_neighbor_stitching_keeps_reading_order(stub_generative):
    rag = _pipeline()
    chunks = rag.vectorstore.chunks
    if len(chunks) < 3:
        pytest.skip("index too small for stitching test")

    anchor = chunks[1]
    stitched = rag._with_neighbors([(anchor, 0.8)])
    texts = [chunk["text"] for chunk, _ in stitched]

    assert anchor["text"] in texts
    # Anchor first (the chunk the reranker scored), then next, then prev.
    expected = [anchor["text"]] + [
        c["text"]
        for c in (chunks[2], chunks[0])
        if c.get("metadata", {}).get("filename") == anchor.get("metadata", {}).get("filename")
    ]
    assert texts == expected
    # All stitched entries carry the anchor's score.
    assert all(score == 0.8 for _, score in stitched)


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_neighbor_stitching_only_top_anchors(stub_generative):
    rag = _pipeline()
    chunks = rag.vectorstore.chunks
    if len(chunks) < 6:
        pytest.skip("index too small for stitching test")

    results = [(chunks[1], 0.9), (chunks[4], 0.5)]
    stitched = rag._with_neighbors(results, max_anchors=1)
    texts = [chunk["text"] for chunk, _ in stitched]
    # Second result is beyond max_anchors: present, but no neighbors added for it.
    assert chunks[4]["text"] in texts
    assert chunks[1]["text"] in texts
    assert len(texts) <= 1 + 2 + 1  # anchor1 + its two neighbors + anchor2
