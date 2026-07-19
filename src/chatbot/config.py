"""
Configuration settings for the RAG system.
All parameters can be adjusted here for different use cases.
"""

import os
from pathlib import Path

# Base paths
# config.py lives at <root>/src/chatbot/config.py, so the project root is three
# levels up. Data lives at <root>/data/ (outside the package). Both can be
# overridden via environment variables for deployment flexibility.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
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
GENERATIVE_GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GENERATIVE_GGUF_FILE = "*q4_k_m.gguf"
# transformers-backend model. 1.5B: reliable, fluent Persian answers. The 0.5B
# variant is ~3x faster but noticeably less accurate.
GENERATIVE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # fast option: "Qwen/Qwen2.5-0.5B-Instruct"
GENERATIVE_MAX_NEW_TOKENS = 350  # cap on answer length (tokens); lower = faster
# Reading the context (prefill) dominates answer latency on CPU: ~42 tok/s,
# and Persian is token-dense (~0.45 tok/char). 2200 chars ≈ 1000 tokens ≈ 24s.
GENERATIVE_MAX_CONTEXT_CHARS = 2200
QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"  # extractive fallback (USE_GENERATIVE=False)
MAX_CONTEXT_LENGTH = 1024  # Maximum context length for QA model (increased)
MAX_ANSWER_LENGTH = 200  # Maximum answer length (increased)

# Logging settings
LOG_LEVEL = "INFO"
LOG_RETRIEVED_CHUNKS = True  # Log retrieved chunks for debugging
LOG_ANSWERS = True  # Log answers for debugging

