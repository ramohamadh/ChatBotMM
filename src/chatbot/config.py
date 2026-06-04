"""
Configuration settings for the RAG system.
All parameters can be adjusted here for different use cases.
"""

import os
from pathlib import Path
from typing import Optional

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
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Multilingual model (supports Persian)
EMBEDDING_DIMENSION = 384  # Dimension for multilingual MiniLM

# Vectorstore settings
VECTORSTORE_INDEX_NAME = "faiss_index"
VECTORSTORE_METADATA_NAME = "metadata.json"

# Retrieval settings
TOP_K = 8  # Number of chunks to retrieve (increased for better context)
HYBRID_SEARCH_ENABLED = True  # Enable hybrid search (semantic + keyword)
KEYWORD_WEIGHT = 0.2  # Weight for keyword search in hybrid (favor semantic)

# QA settings
# Answer engine. Generative (an instruction-tuned LLM) produces fluent, natural
# Persian answers instead of copied fragments. Set USE_GENERATIVE = False to fall
# back to the lightweight extractive model.
USE_GENERATIVE = True
GENERATIVE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # multilingual, good Persian, fits in 8GB on MPS/CPU
QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"  # extractive fallback (USE_GENERATIVE=False)
MAX_CONTEXT_LENGTH = 1024  # Maximum context length for QA model (increased)
MAX_ANSWER_LENGTH = 200  # Maximum answer length (increased)

# Logging settings
LOG_LEVEL = "INFO"
LOG_RETRIEVED_CHUNKS = True  # Log retrieved chunks for debugging
LOG_ANSWERS = True  # Log answers for debugging

