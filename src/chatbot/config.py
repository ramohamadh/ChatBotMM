"""
Configuration settings for the RAG system.
All parameters can be adjusted here for different use cases.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local .env file into os.environ.

    Real environment variables win over the file. Secrets (like the history
    database password) live here so they never enter version control — copy
    .env.example to .env and fill it in.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))

# Base paths
# config.py lives at <root>/src/chatbot/config.py, so the project root is three
# levels up. Data lives at <root>/data/ (outside the package). Both can be
# overridden via environment variables for deployment flexibility.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
_load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = Path(os.environ.get("CHATBOT_DATA_DIR", PROJECT_ROOT / "data"))

DOCS_DIR = Path(os.environ.get("CHATBOT_DOCS_DIR", DATA_DIR / "docs"))
VECTORSTORE_DIR = Path(os.environ.get("CHATBOT_VECTORSTORE_DIR", DATA_DIR / "vectorstore"))
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Document ingestion settings
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# Chunking settings
CHUNK_SIZE = 900  # Number of characters per chunk (longer context for better QA)
CHUNK_OVERLAP = 150  # Overlap between chunks in characters (preserve continuity)

# Embedding settings
# multilingual-e5-base: substantially better Persian retrieval than MiniLM.
# Changing this requires `chatbot rebuild` (the index stores the vectors).
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768  # e5-base dimension (MiniLM was 384)

# Vectorstore settings
VECTORSTORE_INDEX_NAME = "faiss_index"
VECTORSTORE_METADATA_NAME = "metadata.json"

# Retrieval settings
TOP_K = 5  # Number of chunks to retrieve (more = better context but slower answers)
HYBRID_SEARCH_ENABLED = True  # Enable hybrid search (semantic + keyword)
KEYWORD_WEIGHT = 0.2  # Weight for keyword search in hybrid (favor semantic)

# Reranking settings
# A cross-encoder reads (question, chunk) together and judges relevance far
# more precisely than embedding similarity. Retrieval fetches RERANK_CANDIDATES
# chunks, the reranker keeps the RERANK_TOP_K best. Costs ~1-2s per question on
# CPU; the model (~2 GB RAM) downloads once on first use.
RERANK_ENABLED = True
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # multilingual, strong Persian
RERANK_CANDIDATES = 16  # wide net from retrieval; the reranker filters it
RERANK_TOP_K = 6  # chunks passed to the answer model after reranking
# Token cap per (question, chunk) pair. Rerank cost on CPU scales with
# candidates × length² — 320 covers most of a 900-char Persian chunk and is
# ~2x faster than the model's native 512.
RERANK_MAX_LENGTH = 320

# QA settings
# Answer engine. Generative (an instruction-tuned LLM) produces fluent, natural
# Persian answers instead of copied fragments. Set USE_GENERATIVE = False to fall
# back to the lightweight extractive model.
USE_GENERATIVE = True
# Answer engine backend:
#   "llama.cpp"    — quantized 4-bit GGUF model, 2-4x faster on CPU, ~4x less
#                    RAM (recommended; needs the llama-cpp-python package,
#                    falls back to "transformers" automatically if missing)
#   "transformers" — full-precision HuggingFace model
GENERATIVE_BACKEND = "llama.cpp"
# Gemma 3 4B (instruction-tuned): Google's multilingual model (140+ languages,
# strong Persian), 4-bit GGUF (~2.5 GB). Runs on the GPU via Metal on Apple
# Silicon. Chosen over Qwen3-4B after benchmarking on scripts/eval_qa.py.
# Alternatives: "unsloth/gemma-3-12b-it-GGUF" / "gemma-3-12b-it-Q4_K_M.gguf"
# (~7 GB, higher quality, ~2-3x slower) or, for low-RAM machines,
# "Qwen/Qwen2.5-1.5B-Instruct-GGUF" / "*q4_k_m.gguf".
GENERATIVE_GGUF_REPO = "unsloth/gemma-3-4b-it-GGUF"
GENERATIVE_GGUF_FILE = "gemma-3-4b-it-Q4_K_M.gguf"
# transformers-backend model (used only if llama-cpp-python is missing).
# 4B in bf16 needs ~9 GB RAM; on an 8 GB machine use "Qwen/Qwen2.5-1.5B-Instruct".
GENERATIVE_MODEL = "google/gemma-3-4b-it"
GENERATIVE_MAX_NEW_TOKENS = 500  # cap on answer length (tokens); lower = faster
# Sampling temperature. Benchmarked on scripts/eval_qa.py: 0.1 scores no worse
# than 0.3 and rambles less (shorter, faster answers). Raise for chattier
# style at some cost in factual precision.
GENERATIVE_TEMPERATURE = 0.1
# Context cap. Metal-GPU prefill is fast enough that the model, not latency,
# is the bottleneck — 6000 chars ≈ 2700 tokens fits all 6 reranked chunks.
# On a CPU-only machine drop this back to ~2200 (prefill ≈ 42 tok/s there).
GENERATIVE_MAX_CONTEXT_CHARS = 6000
QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"  # extractive fallback (USE_GENERATIVE=False)
MAX_CONTEXT_LENGTH = 1024  # Maximum context length for QA model (increased)
MAX_ANSWER_LENGTH = 200  # Maximum answer length (increased)

# Chat history settings
# Every answered question is recorded (`/history` in chat, `chatbot history`
# in the CLI, GET /history and GET /users on the API). Backend:
#   "postgres" — shared company database; gorest passes user_id/taxpayer_id
#                with each question, so history is per-user. Falls back to
#                SQLite automatically when the server is unreachable.
#   "sqlite"   — local file only (data/history.db).
HISTORY_ENABLED = True
HISTORY_BACKEND = os.environ.get("CHATBOT_HISTORY_BACKEND", "postgres")
HISTORY_DB = Path(os.environ.get("CHATBOT_HISTORY_DB", DATA_DIR / "history.db"))
# Connection settings come from the environment or the gitignored .env file
# (copy .env.example). With no password configured the postgres backend is
# skipped and history falls back to SQLite.
HISTORY_PG = {
    "host": os.environ.get("CHATBOT_PG_HOST", "172.31.0.153"),
    "port": int(os.environ.get("CHATBOT_PG_PORT", "5432")),
    "user": os.environ.get("CHATBOT_PG_USER", "tsp"),
    "password": os.environ.get("CHATBOT_PG_PASSWORD", ""),
    "dbname": os.environ.get("CHATBOT_PG_DBNAME", "ai"),
    "sslmode": os.environ.get("CHATBOT_PG_SSLMODE", "disable"),
    "connect_timeout": int(os.environ.get("CHATBOT_PG_CONNECT_TIMEOUT", "3")),
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_RETRIEVED_CHUNKS = True  # Log retrieved chunks for debugging
LOG_ANSWERS = True  # Log answers for debugging

