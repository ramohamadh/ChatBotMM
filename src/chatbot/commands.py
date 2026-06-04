"""
Core command logic for the RAG system, independent of any CLI framework.

The CLI layer (cli.py) is a thin shell over these functions, which makes the
business logic easy to reuse and test.
"""

import logging
import shutil
import sys
from pathlib import Path

from . import config
from .rag import RAGPipeline

logger = logging.getLogger(__name__)


def get_default_rag_pipeline() -> RAGPipeline:
    """Create and return a RAGPipeline with default multilingual settings."""
    return RAGPipeline(
        docs_directory=str(config.DOCS_DIR),
        vectorstore_directory=str(config.VECTORSTORE_DIR),
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        top_k=config.TOP_K,
        hybrid_search=config.HYBRID_SEARCH_ENABLED,
        keyword_weight=config.KEYWORD_WEIGHT,
        embedding_model=config.EMBEDDING_MODEL,
        qa_model=config.QA_MODEL,
        use_generative=config.USE_GENERATIVE,
        generative_model=config.GENERATIVE_MODEL,
    )


def check_documents() -> list[Path]:
    """Return the list of supported documents in the docs directory."""
    docs_dir = config.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    return (
        list(docs_dir.glob("*.pdf"))
        + list(docs_dir.glob("*.docx"))
        + list(docs_dir.glob("*.txt"))
        + list(docs_dir.glob("*.md"))
    )


def is_indexed() -> bool:
    """True if a FAISS index already exists on disk."""
    return (config.VECTORSTORE_DIR / "faiss_index.index").exists()


def index_documents(force_reindex: bool = False) -> dict:
    """
    Index documents in the docs directory.

    Args:
        force_reindex: If True, delete the existing index and rebuild.

    Returns:
        Indexing statistics dictionary.
    """
    supported_files = check_documents()
    if not supported_files:
        logger.error(f"No documents found in '{config.DOCS_DIR}'")
        logger.info("Please add PDF, DOCX, TXT, or MD files there.")
        raise SystemExit(1)

    logger.info(f"Found {len(supported_files)} document(s) in '{config.DOCS_DIR}'")

    if force_reindex and config.VECTORSTORE_DIR.exists():
        logger.info("Removing existing index...")
        shutil.rmtree(config.VECTORSTORE_DIR)
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Old index removed")

    logger.info("Initializing RAG pipeline...")
    rag = get_default_rag_pipeline()

    logger.info("Indexing documents... (this may take a few minutes)")
    stats = rag.index_documents(force_reindex=force_reindex)

    logger.info("=" * 60)
    logger.info("Indexing complete!")
    logger.info(f"✅ Total documents: {stats.get('total_documents', 0)}")
    logger.info(f"✅ Total chunks: {stats.get('total_chunks', 0)}")
    logger.info(f"✅ Index size: {stats.get('index_size', 0)}")
    logger.info(f"✅ Embedding dimension: {stats.get('embedding_dimension', 384)}")
    logger.info("=" * 60)
    return stats


def _print_answer(response: dict, show_context: bool = False) -> None:
    """Pretty-print an answer response to stdout."""
    print("\n" + "=" * 70)
    print("📝 Answer:")
    print("=" * 70)
    print(response["answer"])
    print()
    if not config.USE_GENERATIVE:
        print(f"🎯 Confidence: {response['score']:.2%}")
    if show_context and response.get("retrieved_chunks"):
        print("\n📄 Retrieved Chunks:")
        for i, chunk in enumerate(response["retrieved_chunks"][:3], 1):
            print(f"\n{i}. Score: {chunk.get('score', 0):.4f}")
            print(f"   {chunk.get('text', '')[:200]}...")
    elif response.get("source_chunks"):
        print("📄 Sources:")
        for i, source in enumerate(response["source_chunks"][:3], 1):
            filename = source.get("metadata", {}).get("filename", "Unknown")
            print(f"   {i}. {filename}")
    print("=" * 70)


def ask_single_question(question: str, return_context: bool = False) -> dict:
    """Ask one question and return the response dict (indexing first if needed)."""
    rag = get_default_rag_pipeline()

    if not is_indexed():
        logger.warning("No index found. Indexing documents first...")
        index_documents()
        rag = get_default_rag_pipeline()

    if not rag.get_stats().get("indexed", False):
        rag.vectorstore.load(rag.vectorstore_directory)
        rag.is_indexed = True

    return rag.ask(question, return_context=return_context)


def interactive_qa(rag: RAGPipeline | None = None) -> None:
    """Start an interactive question-answering session."""
    print("=" * 70)
    print("🤖 RAG System — Interactive Question Answering")
    print("=" * 70)

    try:
        if rag is None:
            print("📚 Loading RAG pipeline...")
            rag = get_default_rag_pipeline()

        stats = rag.get_stats()
        if not stats.get("indexed", False):
            print("\n⚠️  No index found. Indexing documents...")
            index_documents()
            rag = get_default_rag_pipeline()
            stats = rag.get_stats()
            print(f"\n✅ Indexed {stats.get('total_chunks', 0)} chunks")
        else:
            print(f"✅ Loaded existing index ({stats.get('total_chunks', 0)} chunks)")

        print("\n" + "=" * 70)
        print("💬 Ask questions about your documents (Persian or English).")
        print("   Type 'quit', 'exit', or 'q' to stop.")
        print("=" * 70 + "\n")

        while True:
            try:
                question = input("❓ Your question: ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("\n👋 Goodbye!")
                    break
                print("\n🔍 Searching documents...")
                response = rag.ask(question, return_context=False)
                _print_answer(response)
                print()
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                logger.error(f"Error processing question: {e}")
                print(f"\n❌ Error: {e}\n")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Make sure you have documents in '{config.DOCS_DIR}'")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        raise SystemExit(1) from e
