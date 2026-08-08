"""
Cross-encoder reranker.

The bi-encoder retriever embeds the question and the chunks separately, which
is fast but approximate. A cross-encoder reads (question, chunk) together and
judges relevance far more precisely — so retrieval fetches a wide candidate
set and this module picks the few chunks that actually answer the question.
Fewer, better chunks also mean a shorter prefill for the CPU-bound generator.
"""

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Reorders retrieved candidates with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 320,
    ):
        # sentence-transformers is already a hard dependency (embeddings).
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {model_name}")
        # num_labels == 1, so predict() applies a sigmoid and returns
        # probabilities in [0, 1] — directly usable as confidence.
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.model_name = model_name

    def rerank(
        self,
        question: str,
        candidates: list[tuple[dict, float]],
        top_k: int,
    ) -> list[tuple[dict, float]]:
        """Return the `top_k` most relevant (chunk, probability) pairs.

        `candidates` are (chunk, retrieval_score) pairs; the retrieval score
        is discarded in favor of the cross-encoder probability.
        """
        if not candidates:
            return []
        pairs = [(question, chunk.get("text", "")) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(
                (chunk for chunk, _ in candidates),
                (float(s) for s in scores),
                strict=True,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
