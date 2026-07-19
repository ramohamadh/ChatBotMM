"""Tests for the FastAPI REST layer, using a stub pipeline (no models)."""

from fastapi.testclient import TestClient

from chatbot import api


class StubPipeline:
    """Minimal stand-in for RAGPipeline."""

    def __init__(self, indexed=True):
        self.is_indexed = indexed
        self.asked = []

    def ask(self, question, return_context=False, stream_callback=None):
        self.asked.append(question)
        response = {
            "answer": "پایتخت ایران تهران است.",
            "score": 1.0,
            "source_chunks": [
                {"text": "تهران پایتخت ایران است...", "metadata": {"filename": "iran.pdf"}}
            ],
            "timings": {"search_s": 0.01, "answer_s": 0.5},
        }
        if return_context:
            response["retrieved_chunks"] = [
                {"text": "تهران پایتخت ایران است", "metadata": {"filename": "iran.pdf"}, "score": 0.9}
            ]
        return response

    def index_documents(self, force_reindex=False):
        self.is_indexed = True
        return {"total_chunks": 12, "total_documents": 2}

    def get_stats(self):
        return {"indexed": self.is_indexed, "total_chunks": 12 if self.is_indexed else 0}

    def warm_up(self):
        return self


def make_client(indexed=True):
    api.app.state.pipeline = StubPipeline(indexed=indexed)
    return TestClient(api.app)


def test_health():
    client = make_client()
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["indexed"] is True
    assert data["total_chunks"] == 12


def test_ask_returns_answer_and_sources():
    client = make_client()
    response = client.post("/ask", json={"question": "پایتخت ایران کجاست؟"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "پایتخت ایران تهران است."
    assert data["sources"] == ["iran.pdf"]
    assert "retrieved_chunks" not in data  # excluded when return_context is off


def test_ask_with_context():
    client = make_client()
    data = client.post(
        "/ask", json={"question": "سؤال", "return_context": True}
    ).json()
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["metadata"]["filename"] == "iran.pdf"


def test_ask_without_index_is_409():
    client = make_client(indexed=False)
    response = client.post("/ask", json={"question": "سؤال"})
    assert response.status_code == 409


def test_ask_requires_question():
    client = make_client()
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_index_endpoint():
    client = make_client(indexed=False)
    response = client.post("/index", json={"force": True})
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 12
    assert data["total_documents"] == 2
