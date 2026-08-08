"""
RAG Pipeline module.
High-level interface for the complete RAG system.
"""

import logging
import time
from pathlib import Path

from .chunker import (
    TextChunker,
    expand_fragment,
    expand_identifier_question,
    formalize_question,
    normalize_persian,
)
from .embeddings import EmbeddingGenerator
from .extractive_qa import ExtractiveQA
from .generative_qa import GenerativeQA
from .ingestion import DocumentIngester
from .retriever import HybridRetriever
from .vectorstore import FAISSVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class _RefusalGate:
    """Stream buffer that holds text back until it's clearly not a refusal.

    Small models sometimes answer "در سند اطلاعاتی پیدا نشد" even when the
    context contains the answer. The pipeline retries those (see ask()), but
    with streaming the refusal text would already be on the user's screen —
    so pieces are buffered while they are still a prefix of the refusal
    sentence, and only streamed once they diverge from it.
    """

    _REFUSAL = "در سند اطلاعاتی پیدا نشد"

    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""
        self.streaming = False
        self.suppressed = False

    def __call__(self, piece: str) -> None:
        if self.suppressed:
            return
        if self.streaming:
            self.callback(piece)
            return
        self.buffer += piece
        probe = self.buffer.strip()
        if len(probe) < len(self._REFUSAL):
            if self._REFUSAL.startswith(probe):
                return  # still ambiguous — keep buffering
        elif probe.startswith(self._REFUSAL):
            self.suppressed = True  # it IS the refusal — never show it
            return
        # Diverged from the refusal sentence: it's a real answer.
        self.streaming = True
        self.callback(self.buffer)
        self.buffer = ""

    def finish(self) -> None:
        """Flush a short buffered non-refusal answer at the end."""
        if not self.streaming and not self.suppressed and self.buffer:
            self.streaming = True
            self.callback(self.buffer)
            self.buffer = ""


class RAGPipeline:
    """
    High-level RAG pipeline managing all components.

    Usage:
        rag = RAGPipeline("docs/")
        rag.index_documents()
        answer = rag.ask("What is the main topic?")
    """

    def __init__(
        self,
        docs_directory: str = "docs",
        vectorstore_directory: str = "vectorstore",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "intfloat/multilingual-e5-base",
        qa_model: str = "mrm8488/bert-multi-cased-finetuned-xquadv1",
        top_k: int = 5,
        hybrid_search: bool = True,
        keyword_weight: float = 0.3,
        rerank_enabled: bool = True,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_candidates: int = 12,
        rerank_top_k: int = 4,
        rerank_max_length: int = 320,
        use_generative: bool = True,
        generative_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        generative_max_new_tokens: int = 300,
        generative_max_context_chars: int = 3500,
        generative_temperature: float | None = None,
        generative_backend: str = "llama.cpp",
        generative_gguf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        generative_gguf_file: str = "*q4_k_m.gguf",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            docs_directory: Path to directory containing documents
            vectorstore_directory: Path to directory for storing vector index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of sentence-transformers model (default: multilingual)
            qa_model: Name of HuggingFace QA model (default: multilingual)
            top_k: Number of chunks to retrieve
            hybrid_search: Whether to enable hybrid search
            keyword_weight: Weight for keyword search in hybrid mode
        """
        self.docs_directory = Path(docs_directory)
        self.vectorstore_directory = Path(vectorstore_directory)
        self.vectorstore_directory.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        self.ingester = DocumentIngester()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

        self.vectorstore = FAISSVectorStore(
            embedding_dimension=self.embedding_generator.dimension
        )

        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            embedding_generator=self.embedding_generator,
            keyword_weight=keyword_weight,
            enable_hybrid=hybrid_search
        )

        # Answer engine: generative (fluent, chat-like) is the default.
        # The extractive model is kept as a lightweight fallback.
        # Loaded lazily on first ask() — it's by far the heaviest model, and
        # indexing doesn't need it at all.
        self.use_generative = use_generative
        self._generative_model_name = generative_model
        self._generative_max_new_tokens = generative_max_new_tokens
        self._generative_max_context_chars = generative_max_context_chars
        self._generative_temperature = generative_temperature
        self._generative_backend = generative_backend
        self._generative_gguf_repo = generative_gguf_repo
        self._generative_gguf_file = generative_gguf_file
        self._qa_model_name = qa_model
        self._qa = None

        # Cross-encoder reranker: reads (question, chunk) pairs together and
        # picks the truly relevant chunks out of a wide candidate set. Loaded
        # lazily on first ask(), like the answer model.
        self.rerank_enabled = rerank_enabled
        self._rerank_model_name = rerank_model
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k
        self._rerank_max_length = rerank_max_length
        self._reranker = None

        self.top_k = top_k
        self.is_indexed = False

        # Check if index already exists and load it
        index_path = self.vectorstore_directory / "faiss_index.index"
        if index_path.exists():
            try:
                self.vectorstore.load(self.vectorstore_directory)
                self.is_indexed = True
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")

        logger.info("RAG pipeline initialized")

    @property
    def qa(self):
        """The answer engine, created on first use."""
        if self._qa is None:
            if not self.use_generative:
                self._qa = ExtractiveQA(model_name=self._qa_model_name)
                return self._qa

            backend = self._generative_backend
            if backend == "llama.cpp":
                try:
                    from .llama_qa import LlamaGenerativeQA

                    self._qa = LlamaGenerativeQA(
                        repo_id=self._generative_gguf_repo,
                        filename=self._generative_gguf_file,
                        max_new_tokens=self._generative_max_new_tokens,
                        max_context_chars=self._generative_max_context_chars,
                        temperature=self._generative_temperature,
                    )
                    return self._qa
                except ImportError:
                    logger.warning(
                        "llama-cpp-python is not installed — falling back to the "
                        "transformers backend (slower). Install it with: "
                        "pip install llama-cpp-python"
                    )

            self._qa = GenerativeQA(
                model_name=self._generative_model_name,
                max_new_tokens=self._generative_max_new_tokens,
                max_context_chars=self._generative_max_context_chars,
                temperature=self._generative_temperature,
            )
        return self._qa

    @property
    def reranker(self):
        """The cross-encoder reranker, created on first use (or None if disabled)."""
        if not self.rerank_enabled:
            return None
        if self._reranker is None:
            from .reranker import Reranker

            try:
                self._reranker = Reranker(
                    model_name=self._rerank_model_name,
                    max_length=self._rerank_max_length,
                )
            except Exception as e:  # noqa: BLE001 - reranking is an enhancement
                logger.warning(f"Could not load reranker ({e}) — continuing without it")
                self.rerank_enabled = False
                return None
        return self._reranker

    def warm_up(self) -> "RAGPipeline":
        """Force-load the answer model (and reranker) now instead of on the first question."""
        _ = self.qa
        _ = self.reranker
        return self

    # Words that signal a question about the document itself rather than its
    # content. Similarity search cannot answer those (no chunk resembles the
    # question "what is this document about?"), so they get special handling.
    _DOC_WORDS = ("سند", "داکیومنت", "دکیومنت", "دکومنت", "فایل", "متن", "document", "file", "pdf")
    _ABOUT_WORDS = (
        "در مورد", "درباره", "موضوع", "خلاصه", "چیست", "چیه", "جیه", "چی هست",
        "about", "summary", "overview", "topic",
    )

    @classmethod
    def _is_overview_question(cls, question: str) -> bool:
        q = normalize_persian(question).lower()
        return any(w in q for w in cls._DOC_WORDS) and any(w in q for w in cls._ABOUT_WORDS)

    def _intro_chunks(self, per_doc: int = 3) -> list[dict]:
        """The first chunks of each document (title page, purpose section)."""
        by_doc: dict[str, list[dict]] = {}
        for chunk in self.vectorstore.chunks:
            name = chunk.get("metadata", {}).get("filename", "")
            bucket = by_doc.setdefault(name, [])
            if len(bucket) < per_doc:
                bucket.append(chunk)
        return [chunk for chunks in by_doc.values() for chunk in chunks]

    def _with_neighbors(
        self, results: list[tuple[dict, float]], max_anchors: int = 2
    ) -> list[tuple[dict, float]]:
        """Stitch each top chunk together with its physical neighbors.

        Answers often straddle a chunk boundary (the 150-char overlap is not
        always enough). For the best `max_anchors` results, the chunk stored
        directly before/after them in the same document is inserted around
        them, so the generator sees the full passage. The context char budget
        in the answer engine still caps total size.
        """
        chunks = self.vectorstore.chunks
        if not chunks or not results:
            return results
        position = {id(chunk): i for i, chunk in enumerate(chunks)}
        seen_texts = {chunk["text"] for chunk, _ in results}
        stitched: list[tuple[dict, float]] = []
        for rank, (chunk, score) in enumerate(results):
            group = [(chunk, score)]
            i = position.get(id(chunk))
            if rank < max_anchors and i is not None:
                filename = chunk.get("metadata", {}).get("filename")

                def usable(j: int, filename=filename) -> dict | None:
                    if not 0 <= j < len(chunks):
                        return None
                    neighbor = chunks[j]
                    if neighbor.get("metadata", {}).get("filename") != filename:
                        return None
                    if neighbor["text"] in seen_texts:
                        return None
                    return neighbor

                # Anchor stays FIRST: small models weight the leading context
                # chunk far more than the rest, and the anchor is the one the
                # reranker actually scored. Neighbors follow — continuation
                # (next) before predecessor (prev), since an answer cut off
                # mid-chunk continues forward more often than backward.
                if (nxt := usable(i + 1)) is not None:
                    seen_texts.add(nxt["text"])
                    group.append((nxt, score))
                if (prev := usable(i - 1)) is not None:
                    seen_texts.add(prev["text"])
                    group.append((prev, score))
            stitched.extend(group)
        return stitched

    def index_documents(self, force_reindex: bool = False) -> dict:
        """
        Index all documents in the docs directory.

        Args:
            force_reindex: If True, reindex even if index exists

        Returns:
            Dictionary with indexing statistics
        """
        # Check if index already exists
        index_path = self.vectorstore_directory / "faiss_index.index"
        if index_path.exists() and not force_reindex:
            logger.info("Loading existing index...")
            try:
                self.vectorstore.load(self.vectorstore_directory)
                self.is_indexed = True
                stats = self.vectorstore.get_stats()
                logger.info(f"Loaded existing index with {stats['total_chunks']} chunks")
                return stats
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}. Reindexing...")

        # Load documents
        logger.info(f"Loading documents from {self.docs_directory}")
        if not self.docs_directory.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_directory}")

        documents = self.ingester.load_directory(self.docs_directory)

        if not documents:
            logger.warning("No documents found to index")
            return {"total_chunks": 0, "total_documents": 0}

        # Chunk documents
        logger.info("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.warning("No chunks created from documents")
            return {"total_chunks": 0, "total_documents": len(documents)}

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts, show_progress=True)

        # Add to vectorstore
        logger.info("Adding to vectorstore...")
        self.vectorstore.add_embeddings(embeddings, chunks)

        # Save vectorstore
        logger.info("Saving vectorstore...")
        self.vectorstore.save(self.vectorstore_directory)

        self.is_indexed = True

        stats = self.vectorstore.get_stats()
        unique_docs = {metadata.get("filename") for _, metadata in documents if metadata.get("filename")}
        stats["total_documents"] = len(unique_docs) if unique_docs else len(documents)
        logger.info(f"Indexing complete: {stats['total_chunks']} chunks from {len(documents)} documents")

        return stats

    def ask(self, question: str, return_context: bool = False, stream_callback=None) -> dict:
        """
        Ask a question and get an answer from the indexed documents.

        Args:
            question: The question to ask
            return_context: If True, include retrieved chunks in response
            stream_callback: Optional callable receiving each generated text
                piece as it is produced (generative engine only)

        Returns:
            Dictionary with:
            - answer: The extracted answer
            - score: Confidence score
            - retrieved_chunks: List of retrieved chunks (if return_context=True)
            - source_chunks: Source information for the answer
        """
        if not self.is_indexed:
            raise ValueError("Documents must be indexed first. Call index_documents()")

        logger.info(f"Processing question: {question}")

        # Colloquial forms ("هارو", "چیه") reliably confuse small models into
        # refusing even with the answer in context; formalize first, and turn
        # bare noun-phrase fragments into explicit requests.
        question = expand_fragment(
            expand_identifier_question(formalize_question(question))
        )

        # Retrieve relevant chunks. With reranking, cast a wide net first —
        # the cross-encoder decides which candidates actually answer the
        # question far better than embedding similarity can.
        search_started = time.time()
        reranker = self.reranker
        if reranker is not None:
            candidates = self.retriever.retrieve(question, top_k=self.rerank_candidates)
            retrieved = reranker.rerank(question, candidates, top_k=self.rerank_top_k)

            # Exact-identifier safety net: the retriever guarantees that
            # chunks literally containing a rare query token (field codes
            # like "ins") lead its results, but the cross-encoder is free to
            # rank them out of the top-k again. If that happened, re-insert
            # the retriever's best rare chunk right after the reranker's top
            # pick.
            rare_ids = getattr(self.retriever, "last_rare_chunk_ids", set())
            if rare_ids and not any(id(c) in rare_ids for c, _ in retrieved):
                rare_best = next(
                    ((c, s) for c, s in candidates if id(c) in rare_ids), None
                )
                if rare_best is not None:
                    logger.info("Reranker dropped all rare-token chunks — re-inserting one")
                    retrieved = (
                        retrieved[:1] + [rare_best] + retrieved[1:]
                    )[: self.rerank_top_k]
        else:
            retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        retrieved = self._with_neighbors(retrieved)

        overview_injected = False
        if self._is_overview_question(question):
            # "What is this document about?" — the answer lives in the
            # document's opening (title / هدف section), which similarity
            # search never surfaces. Put those chunks first.
            intro = self._intro_chunks()
            intro_texts = {chunk["text"] for chunk in intro}
            retrieved = [(chunk, 1.0) for chunk in intro] + [
                (chunk, score) for chunk, score in retrieved if chunk["text"] not in intro_texts
            ]
            retrieved = retrieved[: self.top_k + 3]
            overview_injected = True

        if not retrieved:
            return {
                "answer": "No relevant information found in the documents.",
                "score": 0.0,
                "retrieved_chunks": [],
                "source_chunks": []
            }

        # Log retrieved chunks for debugging
        logger.info(f"Retrieved {len(retrieved)} chunks:")
        for i, (chunk, score) in enumerate(retrieved[:3], 1):
            logger.debug(f"  {i}. Score: {score:.4f} - {chunk['text'][:150]}...")

        search_seconds = time.time() - search_started

        # Generate (or extract) an answer from the retrieved chunks.
        answer_started = time.time()
        chunks_list = [chunk for chunk, _ in retrieved]
        if self.use_generative:
            # Only pass stream_callback when set, so stubs/tests with the plain
            # (question, chunks) signature keep working.
            if stream_callback is not None:
                gate = _RefusalGate(stream_callback)
                qa_result = self.qa.answer(question, chunks_list, stream_callback=gate)
            else:
                gate = None
                qa_result = self.qa.answer(question, chunks_list)

            # False-refusal retry: "not found" despite a strong retrieval hit
            # almost always means the model misread the question, not that the
            # answer is missing. One retry with an explicit nudge fixes most
            # of these (costs one extra generation only in this rare case).
            # Rare-token questions ("ins چیست") retry regardless of score:
            # cross-encoder scores run low on bare identifiers even when the
            # right table row IS in context, and the verbatim retry is exactly
            # what quoting a table/JSON answer needs.
            top_score = retrieved[0][1] if retrieved else 0.0
            has_rare = bool(getattr(self.retriever, "last_rare_chunk_ids", set()))
            refusal_marker = "پیدا نشد"
            if refusal_marker in qa_result.get("answer", "") and (
                top_score >= 0.55 or has_rare
            ):
                logger.info("Refusal despite strong retrieval — retrying with a nudge")
                nudged = (
                    f"{question}\n"
                    "(اطلاعات مرتبط با این سؤال در متن زمینه وجود دارد؛ "
                    "آن را از متن استخراج کن.)"
                )
                retry_gate = _RefusalGate(stream_callback) if stream_callback else None
                # verbatim=True: refusals often happen because the answer must
                # be quoted token-for-token (JSON samples, tables) and the
                # repetition penalty fights exactly that.
                if retry_gate is not None:
                    retry_result = self.qa.answer(
                        nudged, chunks_list, stream_callback=retry_gate, verbatim=True
                    )
                else:
                    retry_result = self.qa.answer(nudged, chunks_list, verbatim=True)
                if refusal_marker not in retry_result.get("answer", ""):
                    qa_result = retry_result
                    gate = retry_gate
            if gate is not None and refusal_marker not in qa_result.get("answer", ""):
                gate.finish()
        else:
            qa_result = self.qa.extract_from_chunks(question, chunks_list)

        logger.info(f"QA result - Score: {qa_result['score']:.3f}, Answer length: {len(qa_result.get('answer', ''))}")

        # Heuristic confidence from retrieval quality: the generative model
        # has no calibrated self-confidence, but the top retrieval score
        # tracks answer reliability well in practice. With the reranker, the
        # cross-encoder probability is used directly (it is far better
        # calibrated than embedding similarity).
        # Overview questions get intro chunks with a hand-assigned score of
        # 1.0 — a confidence derived from that would always read ~99% while
        # meaning nothing, so none is shown at all.
        confidence: int | None = None
        if (
            retrieved
            and not overview_injected
            and "پیدا نشد" not in qa_result.get("answer", "")
        ):
            top = retrieved[0][1]
            if reranker is not None:
                confidence = round(min(max(top, 0.0), 1.0) * 94 + 5)
            else:
                confidence = round(min(max((top - 0.30) / (0.85 - 0.30), 0.0), 1.0) * 80 + 15)

        # Prepare response
        response = {
            "answer": qa_result["answer"],
            "score": qa_result["score"],
            "confidence": confidence,
            "source_chunks": qa_result.get("source_chunks", []),
            "timings": {
                "search_s": search_seconds,
                "answer_s": time.time() - answer_started,
            },
        }

        if return_context:
            response["retrieved_chunks"] = [
                {
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": score
                }
                for chunk, score in retrieved
            ]

        # Log answer if enabled
        if hasattr(self, 'log_answers') and self.log_answers:
            logger.info(f"Answer: {response['answer']}")
            logger.info(f"Confidence score: {response['score']:.4f}")

        return response

    def add_documents(self, documents: list[tuple]) -> dict:
        """
        Incrementally add new documents to the existing index.

        Args:
            documents: List of (text, metadata) tuples

        Returns:
            Dictionary with statistics
        """
        if not self.is_indexed:
            logger.warning("No existing index. Use index_documents() instead.")
            return self.index_documents()

        logger.info(f"Adding {len(documents)} new documents to index...")

        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            return {"added_chunks": 0, "added_documents": len(documents)}

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Add to vectorstore
        self.vectorstore.add_embeddings(embeddings, chunks)

        # Save updated vectorstore
        self.vectorstore.save(self.vectorstore_directory)

        stats = {
            "added_chunks": len(chunks),
            "added_documents": len(documents),
            "total_chunks": len(self.vectorstore.chunks)
        }

        logger.info(f"Added {len(chunks)} chunks. Total chunks: {stats['total_chunks']}")
        return stats

    def get_stats(self) -> dict:
        """Get statistics about the indexed documents."""
        if not self.is_indexed:
            return {"indexed": False}

        stats = self.vectorstore.get_stats()
        stats["indexed"] = True
        return stats

