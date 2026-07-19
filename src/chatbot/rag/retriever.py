"""
Retriever module.
Implements hybrid search combining semantic and keyword-based retrieval.
"""

import logging
import math
import re
from collections import Counter

from .chunker import normalize_persian

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""

    def __init__(
        self,
        vectorstore,
        embedding_generator,
        keyword_weight: float = 0.3,
        enable_hybrid: bool = True
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vectorstore: FAISSVectorStore instance
            embedding_generator: EmbeddingGenerator instance
            keyword_weight: Weight for keyword search (0.0 = semantic only, 1.0 = keyword only)
            enable_hybrid: Whether to enable hybrid search (if False, uses semantic only)
        """
        self.vectorstore = vectorstore
        self.embedding_generator = embedding_generator
        self.keyword_weight = keyword_weight
        self.enable_hybrid = enable_hybrid

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk_dict, combined_score) tuples, sorted by score
        """
        # Normalize the query the same way indexed text was normalized.
        query = normalize_persian(query)

        # Semantic search
        query_embedding = self.embedding_generator.generate_embedding(query)
        semantic_results = self.vectorstore.search(query_embedding, top_k=top_k * 2)  # Get more for hybrid

        if not self.enable_hybrid:
            return semantic_results[:top_k]

        # Keyword scores over the WHOLE corpus, not just the semantic hits.
        # Rare exact terms (field codes like "ins", "inp", "tinb") often have
        # weak embedding similarity, so keyword-only matches must be able to
        # enter the candidate set on their own.
        corpus = getattr(self.vectorstore, "chunks", None) or [c for c, _ in semantic_results]
        keyword_scores, rare_chunk_ids = self._keyword_scores(query, corpus)

        candidates: list[tuple[dict, float]] = list(semantic_results)
        semantic_ids = {id(chunk) for chunk, _ in semantic_results}
        min_semantic = min((score for _, score in semantic_results), default=0.0)
        chunk_by_id = {id(chunk): chunk for chunk in corpus}
        keyword_only = sorted(
            (
                (chunk_id, score)
                for chunk_id, score in keyword_scores.items()
                if score > 0 and chunk_id not in semantic_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: top_k * 2]
        for chunk_id, _ in keyword_only:
            # Semantic score unknown for these; use the weakest semantic score
            # as a floor so a strong keyword match can still outrank weak hits.
            candidates.append((chunk_by_id[chunk_id], min_semantic))

        combined_results = [
            (
                chunk,
                (1 - self.keyword_weight) * semantic_score
                + self.keyword_weight * keyword_scores.get(id(chunk), 0.0),
            )
            for chunk, semantic_score in candidates
        ]
        combined_results.sort(key=lambda x: x[1], reverse=True)
        results = combined_results[:top_k]

        # Exact-identifier guarantee: when the query names a rare token (a
        # field code such as "ins"), the chunks that literally contain it MUST
        # lead the results. Score blending alone cannot ensure this: common
        # query words (فیلد، اجباری…) give unrelated pages high similarity and
        # push the only relevant chunks down or out — and small models weigh
        # the first context chunks much more heavily than the rest.
        if rare_chunk_ids:
            rare_in_results = [item for item in results if id(item[0]) in rare_chunk_ids]
            if not rare_in_results:
                rare_in_results = sorted(
                    (item for item in combined_results if id(item[0]) in rare_chunk_ids),
                    key=lambda item: keyword_scores.get(id(item[0]), 0.0),
                    reverse=True,
                )[:3]
            if rare_in_results:
                others = [item for item in results if id(item[0]) not in rare_chunk_ids]
                results = rare_in_results + others[: max(top_k - len(rare_in_results), 0)]

        return results

    # A query token appearing in at most this many chunks is treated as an
    # exact identifier (field code, section number) rather than a normal word.
    RARE_TOKEN_MAX_DF = 5

    def _tokenized_corpus(self, corpus: list[dict]) -> list[tuple[dict, Counter]]:
        """Tokenize the corpus once and cache it (it only changes on reindex)."""
        key = (id(corpus), len(corpus))
        if getattr(self, "_token_cache_key", None) != key:
            self._token_cache_key = key
            self._token_cache = [
                (chunk, Counter(self._tokenize(chunk["text"].lower()))) for chunk in corpus
            ]
        return self._token_cache

    def _keyword_scores(self, query: str, chunks: list[dict]):
        """
        Term-frequency keyword scores for every chunk, keyed by id(chunk).

        Returns (scores, rare_chunk_ids): scores are length-normalized and
        scaled to [0, 1]; rare_chunk_ids is the set of chunks containing any
        rare query token (df <= RARE_TOKEN_MAX_DF).
        """
        query_tf = Counter(self._tokenize(query.lower()))
        tokenized = self._tokenized_corpus(chunks)

        # IDF: a query token found in 3 chunks (a field code like "ins") must
        # vastly outweigh one found on every page (e.g. "is" from the
        # "RC_IITP.IS" header). Squared, because linear IDF is not enough:
        # فیلد/اجباری appear on every table page and their combined term
        # frequency would still outvote the one identifying token.
        total = len(tokenized) or 1
        idf: dict[str, float] = {}
        rare_chunk_ids: set = set()
        for token in query_tf:
            containing = [id(chunk) for chunk, chunk_tf in tokenized if token in chunk_tf]
            df = len(containing)
            idf[token] = (math.log((total + 1) / (df + 1)) + 1e-6) ** 2
            if 0 < df <= self.RARE_TOKEN_MAX_DF:
                rare_chunk_ids.update(containing)

        scores: dict[int, float] = {}
        for chunk, chunk_tf in tokenized:
            token_count = sum(chunk_tf.values())
            if not token_count:
                continue
            score = sum(
                query_count * chunk_tf[token] * idf[token]
                for token, query_count in query_tf.items()
                if token in chunk_tf
            )
            if score > 0:
                scores[id(chunk)] = score / token_count

        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores, rare_chunk_ids

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization (split on whitespace and punctuation).
        Supports Persian/Farsi text better.
        """
        # Normalize Persian/Arabic characters (shared with indexing path)
        text = normalize_persian(text)

        # Remove punctuation but keep Persian/Arabic characters
        # Keep alphanumeric, Persian/Arabic, and whitespace
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F]', ' ', text)

        tokens = text.split()
        # Filter out very short tokens (but keep Persian words which can be shorter)
        return [t for t in tokens if len(t) > 1]

