# ChatBotMM — Local Persian/English RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) system in Python. It ingests
documents, embeds them, stores them in a FAISS vector index, and answers
questions using a **local generative LLM** — producing fluent, natural answers.

**Supports both Persian/Farsi and English!** 🇮🇷 🇬🇧

> 🇮🇷 نسخه‌ی فارسی این راهنما: [FAREADME.md](FAREADME.md)

## Features

- **Document Ingestion**: PDF, DOCX, TXT, and MD files
- **Smart Chunking**: Configurable chunk size/overlap, with Persian normalization
- **Multilingual Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages incl. Persian)
- **Vector Storage**: FAISS with save/load and incremental updates
- **Hybrid Retrieval**: Semantic + keyword search
- **Generative Answers**: Local instruction-tuned LLM (`Qwen2.5-0.5B-Instruct` by default, fast on CPU) writes fluent Persian — not just copied fragments, streamed live into the terminal. An extractive fallback is also available.
- **Packaged**: `src/` layout, `pyproject.toml`, console entrypoint, and tests

## Installation

```bash
# from the project root
python -m venv venv && source venv/bin/activate
pip install -e .            # installs the package + the `chatbot` command
# for development (tests, linters):
pip install -e ".[dev]"
```

Place your documents in `data/docs/` (PDF, DOCX, TXT, or MD).

## Quick Start

The fastest way — one command that installs deps, indexes, and starts chatting:

```bash
python main.py cli      # or, if installed: chatbot cli
```

`cli` is idempotent: it installs requirements only if missing, indexes only if
there's no index, then drops you into interactive chat.

Or run the individual commands:

```bash
# 1. Index the documents in data/docs/ -> data/vectorstore/
chatbot index

# 2. Ask questions (Persian or English)
chatbot                                      # interactive chat (type questions, 'quit' to stop)
chatbot ask "این سند درباره چیست؟"           # single question
chatbot ask "What is this about?" --context  # include retrieved chunks

# Rebuild the index from scratch
chatbot rebuild        # or: chatbot index --force
```

> Every command also works without installing, via the source-checkout shim
> `python main.py <command>` or as a module `python -m chatbot <command>`.
> The CLI is built with [Typer](https://typer.tiangolo.com/).

> **First run downloads the generative model (~3 GB) from HuggingFace** and needs
> internet once. After that it runs fully offline. See [USAGE_FA.md](USAGE_FA.md).

## Commands

| Command | What it does |
| --- | --- |
| `chatbot` | Interactive chat — ask as many questions as you like in one session. |
| `chatbot cli` | One-shot bootstrap: install deps → index → chat (each step skipped if done). Use `--skip-install` to skip the install step. |
| `chatbot index` | Index `data/docs/` into the vector store. `--force` / `-f` rebuilds. |
| `chatbot rebuild` | Rebuild the index from scratch (alias for `index --force`). |
| `chatbot ask [QUESTION]` | Ask a question; omit `QUESTION` for interactive mode. `--context` / `-c` shows the retrieved chunks. |

Run `chatbot --help` or `chatbot <command> --help` for full details.

## Docker

```bash
# build (or pull the image published by CI: ghcr.io/<owner>/<repo>)
docker build -t chatbotmm .

# copy your documents into the data volume, then chat
docker run --rm -v chatbot-data:/data -v ./data/docs:/src alpine sh -c "mkdir -p /data/docs && cp -r /src/. /data/docs/"
docker run -it -v chatbot-data:/data chatbotmm
```

All persistent state (documents, index, downloaded models) lives in the
`chatbot-data` volume; models (~2 GB) download once on first run.
CI (GitHub Actions) lints, tests, and publishes the image to GHCR on every
push to `main` — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Usage in Python

```python
from chatbot import RAGPipeline

rag = RAGPipeline(docs_directory="data/docs", vectorstore_directory="data/vectorstore")
rag.index_documents()

response = rag.ask("موضوع اصلی سند چیست؟")
print(response["answer"])
print(f"Sources: {len(response['source_chunks'])} chunks")
```

## Project Structure

```
.
├── main.py                 # source-checkout shim: `python main.py <command>`
├── pyproject.toml          # packaging, console entrypoint, ruff/black/pytest config
├── requirements.txt        # pinned deps (also declared in pyproject)
├── README.md  FAREADME.md  USAGE_FA.md
├── data/
│   ├── docs/               # put your documents here
│   └── vectorstore/        # generated FAISS index (gitignored)
├── src/chatbot/
│   ├── __init__.py         # exports RAGPipeline, __version__
│   ├── __main__.py         # enables `python -m chatbot`
│   ├── config.py           # configuration (paths overridable via env vars)
│   ├── cli.py              # thin Typer CLI (the `chatbot` command)
│   ├── commands.py         # framework-agnostic command logic
│   ├── bootstrap.py        # dependency auto-install + re-exec
│   └── rag/
│       ├── ingestion.py    # document loading
│       ├── chunker.py      # chunking + normalize_persian()
│       ├── embeddings.py   # embedding generation
│       ├── vectorstore.py  # FAISS storage
│       ├── retriever.py    # hybrid retrieval
│       ├── generative_qa.py# local generative LLM answering (default)
│       ├── extractive_qa.py# extractive fallback
│       └── pipeline.py     # high-level RAGPipeline
└── tests/                  # pytest suite (chunker, config, generative, pipeline, cli)
```

## Configuration

Edit [src/chatbot/config.py](src/chatbot/config.py):

- `USE_GENERATIVE` — `True` (default) for fluent generated answers; `False` for the extractive fallback
- `GENERATIVE_MODEL` — default `Qwen/Qwen2.5-0.5B-Instruct` (fast CPU answers); switch to `Qwen/Qwen2.5-1.5B-Instruct` for higher quality (~3x slower on CPU)
- `GENERATIVE_MAX_NEW_TOKENS`, `GENERATIVE_MAX_CONTEXT_CHARS` — answer length / context size caps (lower = faster)
- `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, hybrid-search weights

Data locations can be overridden with environment variables:
`CHATBOT_DATA_DIR`, `CHATBOT_DOCS_DIR`, `CHATBOT_VECTORSTORE_DIR`.

## How It Works

1. **Ingest** documents from `data/docs/`, extracting text + metadata
2. **Chunk** into overlapping pieces (with Persian normalization)
3. **Embed** with multilingual sentence-transformers (local)
4. **Store** embeddings in a FAISS index
5. **Retrieve** relevant chunks via hybrid (semantic + keyword) search
6. **Generate** a grounded answer with a local instruction-tuned LLM

## Models

- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (384-d, 50+ languages)
- **Generation**: `Qwen/Qwen2.5-0.5B-Instruct` (multilingual, fast on CPU; `1.5B` optional for quality)
- **Extractive fallback**: `mrm8488/bert-multi-cased-finetuned-xquadv1`

Models download automatically on first use and are cached in `~/.cache/huggingface/`.

## Development

```bash
pip install -e ".[dev]"
pytest          # run the test suite
ruff check .    # lint
black .         # format
```

## Troubleshooting

- **Poor/garbled Persian answers** — make sure `USE_GENERATIVE = True`. If you have an old index from a previous version, run `chatbot rebuild`.
- **Index not found** — run `chatbot index` first.
- **Out of memory loading the model** — switch `GENERATIVE_MODEL` to `Qwen/Qwen2.5-0.5B-Instruct`.
- **First run is slow** — the generative model (~3 GB) is downloading; subsequent runs are fast and offline.

## License

MIT — provided as-is for educational and production use.
