"""
Core command logic for the RAG system, independent of any CLI framework.

The CLI layer (cli.py) is a thin shell over these functions, which makes the
business logic easy to reuse and test.
"""

import logging
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config
from .errors import console, print_friendly_error
from .rag import RAGPipeline

logger = logging.getLogger(__name__)


def _quiet_logs() -> None:
    """Hide routine log chatter so users only see friendly output.

    Everything below WARNING is hidden globally; the HF-hub/HTTP loggers are
    pushed to ERROR because they emit routine WARNINGs too (e.g. the anonymous
    rate-limit notice). Model-download progress bars are unaffected — they are
    drawn directly to the terminal, not logged.
    """
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("huggingface_hub", "httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Python warnings are a separate channel from logging — third-party libs
    # emit deprecation/user warnings (e.g. llama-cpp-python passing deprecated
    # HF arguments) that mean nothing to end users.
    for category in (UserWarning, FutureWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=category)
    # Hide transformers' internal "Loading weights" tqdm bar — it flashes by in
    # under a second and renders as ASCII noise in some terminals. Our rich
    # spinner covers that phase; first-run *download* bars (huggingface_hub)
    # are separate and stay visible.
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - purely cosmetic, never fail on it
        pass


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
        generative_max_new_tokens=config.GENERATIVE_MAX_NEW_TOKENS,
        generative_max_context_chars=config.GENERATIVE_MAX_CONTEXT_CHARS,
        generative_backend=config.GENERATIVE_BACKEND,
        generative_gguf_repo=config.GENERATIVE_GGUF_REPO,
        generative_gguf_file=config.GENERATIVE_GGUF_FILE,
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


def index_documents(force_reindex: bool = False, rag: RAGPipeline | None = None) -> dict:
    """
    Index documents in the docs directory.

    Args:
        force_reindex: If True, delete the existing index and rebuild.
        rag: An already-constructed pipeline to reuse. Loading the models takes
            minutes and gigabytes of RAM, so callers that already have a
            pipeline should pass it in instead of letting us build a new one.

    Returns:
        Indexing statistics dictionary.
    """
    supported_files = check_documents()
    if not supported_files:
        console.print(f"[red]❌ No documents found in '{config.DOCS_DIR}'.[/red]")
        console.print("💡 Add PDF, DOCX, TXT, or MD files there, then run 'chatbot index'.")
        raise SystemExit(1)

    console.print(f"📄 Found {len(supported_files)} document(s) in '{config.DOCS_DIR}'")

    if force_reindex and config.VECTORSTORE_DIR.exists():
        console.print("🗑  Removing the old index…")
        shutil.rmtree(config.VECTORSTORE_DIR)
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if rag is None:
        console.print("📚 Loading models… (the first run downloads them — progress below)")
        rag = get_default_rag_pipeline()

    console.print("⚙️  Indexing documents… (progress below)")
    stats = rag.index_documents(force_reindex=force_reindex)

    console.print(
        Panel.fit(
            f"[green]✅ Indexed {stats.get('total_documents', 0)} document(s) — "
            f"{stats.get('total_chunks', 0)} chunks "
            f"(embedding dim {stats.get('embedding_dimension', 384)})[/green]",
            border_style="green",
        )
    )
    return stats


def _answer_panel(text: str) -> Panel:
    """The green answer panel; also used as the live-updating streaming frame."""
    body = Text(text) if text else Text("🤔 Thinking…", style="dim")
    return Panel(
        body,
        title="📝 Answer",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    )


def _print_answer(response: dict, show_context: bool = False, panel: bool = True) -> None:
    """Pretty-print an answer response using rich panels.

    With panel=False only the metadata (confidence/sources/chunks) is printed —
    used after streaming, where the answer panel is already on screen.
    """
    if panel:
        console.print()
        console.print(_answer_panel(response["answer"]))
    if not config.USE_GENERATIVE:
        console.print(f"🎯 Confidence: {response['score']:.2%}")
    if show_context and response.get("retrieved_chunks"):
        console.print(Text("📄 Retrieved chunks:", style="bold"))
        for i, chunk in enumerate(response["retrieved_chunks"][:3], 1):
            snippet = chunk.get("text", "")[:200]
            console.print(
                Text(f"  {i}. score {chunk.get('score', 0):.4f} — {snippet}...", style="dim")
            )
    elif response.get("source_chunks"):
        labels: list[str] = []
        for source in response["source_chunks"][:3]:
            metadata = source.get("metadata", {})
            label = metadata.get("filename", "Unknown")
            page = metadata.get("page_number")
            if page:
                label += f" (page {page})"
            if label not in labels:
                labels.append(label)
        console.print(Text("📄 " + "  |  ".join(labels), style="dim"))


def ask_single_question(question: str, return_context: bool = False) -> dict:
    """Ask one question and return the response dict (indexing first if needed)."""
    _quiet_logs()
    console.print("📚 Loading models… (the first run downloads them — progress below)")
    rag = get_default_rag_pipeline()

    if not rag.is_indexed:
        console.print("⚠️  No index found — indexing documents first…")
        index_documents(rag=rag)

    console.print(Text("🔍 Searching & thinking…", style="dim yellow"))
    return rag.ask(question, return_context=return_context)


def _flush_typeahead() -> None:
    """Discard keystrokes typed while the model was busy generating.

    Generating an answer can take minutes on CPU; stray Enter presses during
    that time would otherwise be consumed as empty questions, reprinting the
    prompt once per keypress. Called only AFTER an answer completes — never
    before reading input — so a question typed early (e.g. while the models
    are still loading) is kept, and a multi-byte character can't be cut in
    half right before input() decodes it.
    """
    if sys.stdin.isatty():
        try:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (ImportError, OSError):
            pass  # non-POSIX platform — nothing to flush


def _read_question() -> str:
    """Prompt for the next question.

    UI text stays English/symbols: terminals handle right-to-left text badly
    when it's mixed into prompts (cursor lands on the wrong side). The
    *answers* are still Persian — that's document content, rendered as-is.
    """
    return console.input("[bold green]👤 You ❯ [/]").strip()


def _hf_model_cached(repo_id: str) -> bool:
    """True if a HuggingFace repo already has a snapshot in the local cache.

    Used to decide between a spinner (cached: nothing else draws on screen)
    and a static message (downloading: the download's own progress bars need
    the terminal — a spinner redrawing its line would shred them).
    """
    if "/" not in repo_id:
        repo_id = f"sentence-transformers/{repo_id}"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def _load_pipeline_with_status() -> RAGPipeline:
    """Load the pipeline with staged status output.

    All models cached  -> animated spinner per stage, then "✔ Ready".
    Downloads expected -> static lines; the download bars show the progress.
    """
    answer_model_cached = (
        _hf_model_cached(config.GENERATIVE_GGUF_REPO)
        if config.GENERATIVE_BACKEND == "llama.cpp"
        else _hf_model_cached(config.GENERATIVE_MODEL)
    )
    if not (_hf_model_cached(config.EMBEDDING_MODEL) and answer_model_cached):
        console.print(
            "[yellow]📚 Loading models… (the first run downloads them — progress below)[/yellow]"
        )
        rag = get_default_rag_pipeline()
        console.print("[yellow]🧠 Loading the answer model…[/yellow]")
        rag.warm_up()
        return rag

    with console.status("[yellow]Loading embedding model…[/yellow]", spinner="dots") as status:
        rag = get_default_rag_pipeline()
        status.update("[yellow]Loading the answer model…[/yellow]")
        rag.warm_up()
    console.print("[green]✔ Ready[/green]")
    return rag


def _session_panel(rag: RAGPipeline, stats: dict) -> Panel:
    """Startup summary: models, corpus size, device."""
    qa_backend = type(rag.qa).__name__ if getattr(rag, "_qa", None) is not None else "?"
    if qa_backend == "LlamaGenerativeQA":
        model_label = f"{config.GENERATIVE_GGUF_REPO.split('/')[-1]} (quantized, llama.cpp)"
    else:
        model_label = config.GENERATIVE_MODEL.split("/")[-1]
    documents = len(
        {
            chunk.get("metadata", {}).get("filename", "?")
            for chunk in getattr(rag.vectorstore, "chunks", [])
        }
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan")
    grid.add_column()
    grid.add_row("Model", model_label)
    grid.add_row("Embedding", config.EMBEDDING_MODEL.split("/")[-1])
    grid.add_row("Documents", str(documents))
    grid.add_row("Chunks", str(stats.get("total_chunks", 0)))
    grid.add_row("Device", "CPU")
    return Panel(grid, border_style="cyan", title="Session", title_align="left")


def _print_suggestions(rag: RAGPipeline) -> None:
    """A few example questions so first-time users know what to ask."""
    headings: list[str] = []
    for chunk in getattr(rag.vectorstore, "chunks", []):
        for line in chunk.get("text", "").splitlines():
            match = re.match(r"^\s*\d+(?:[-–]\d+){0,3}[-–]?\s+(\S.{3,60})$", line)
            if match:
                headings.append(match.group(1).strip())
                if len(headings) >= 2:
                    break
        if len(headings) >= 2:
            break

    console.print("[cyan]💡 Try asking:[/cyan]")
    console.print(Text("  • این سند در مورد چیست؟", style="dim"))
    for heading in headings:
        console.print(Text(f"  • دربارهٔ «{heading}» توضیح بده", style="dim"))
    console.print(Text("  • /help for commands", style="dim"))
    console.print()


def _print_confidence(response: dict) -> None:
    """Retrieval-based confidence: how well the documents matched the question."""
    confidence = response.get("confidence")
    if confidence is None:
        return
    if confidence >= 70:
        icon, style = "🟢", "green"
    elif confidence >= 40:
        icon, style = "🟡", "yellow"
    else:
        icon, style = "🔴", "red"
    console.print(Text(f"{icon} Confidence: {confidence}%", style=style))


def _print_timings(response: dict, total_s: float) -> None:
    timings = response.get("timings", {})
    parts = []
    if "search_s" in timings:
        parts.append(f"Search {timings['search_s'] * 1000:.0f} ms")
    if "answer_s" in timings:
        parts.append(f"Generation {timings['answer_s']:.2f} s")
    parts.append(f"Total {total_s:.2f} s")
    console.print(Text("⏱  " + "  ·  ".join(parts), style="dim"))


_HELP_TEXT = """\
[cyan]/help[/cyan]      show this help
[cyan]/stats[/cyan]     session info (models, documents, chunks)
[cyan]/context[/cyan]   show the chunks retrieved for the last answer
[cyan]/sources[/cyan]   show the sources of the last answer
[cyan]/history[/cyan]   list the questions asked this session
[cyan]/reindex[/cyan]   rebuild the index from data/docs
[cyan]/clear[/cyan]     clear the screen
[cyan]/exit[/cyan]      quit (also: quit, exit, q, Ctrl+C)"""


def _handle_command(
    command: str,
    rag: RAGPipeline,
    history: list[str],
    last_response: dict | None,
) -> bool:
    """Handle a /command; returns True when the session should end."""
    name = command.lower().lstrip("/")

    if name in ("exit", "quit", "q"):
        console.print("\n👋 [bold]Goodbye![/bold]")
        return True
    if name == "help":
        console.print(Panel(_HELP_TEXT, border_style="cyan", title="Commands", title_align="left"))
    elif name == "stats":
        console.print(_session_panel(rag, rag.get_stats()))
    elif name == "context":
        chunks = (last_response or {}).get("retrieved_chunks") or []
        if not chunks:
            console.print("[yellow]No retrieved context yet — ask a question first.[/yellow]")
        for i, chunk in enumerate(chunks, 1):
            page = chunk.get("metadata", {}).get("page_number", "?")
            snippet = chunk.get("text", "")[:150].replace("\n", " ")
            console.print(f"[cyan]{i}. page {page}[/cyan] [dim](score {chunk.get('score', 0):.3f})[/dim]")
            console.print(Text(f"   {snippet}…", style="dim"))
    elif name == "sources":
        if last_response:
            _print_answer(last_response, panel=False)
        else:
            console.print("[yellow]No answer yet — ask a question first.[/yellow]")
    elif name == "history":
        if not history:
            console.print("[yellow]No questions asked yet.[/yellow]")
        for i, question in enumerate(history, 1):
            console.print(f"[cyan]{i}.[/cyan] {question}")
    elif name == "clear":
        console.clear()
    elif name == "reindex":
        index_documents(force_reindex=True, rag=rag)
    else:
        console.print(f"[yellow]Unknown command '{command}' — try /help.[/yellow]")
    return False


def interactive_qa(rag: RAGPipeline | None = None) -> None:
    """Start an interactive question-answering session."""
    _quiet_logs()
    console.print(
        Panel.fit(
            "[bold cyan]🤖 ChatBotMM[/bold cyan]\n"
            "Chat with your documents, in Persian or English.\n"
            "[dim]/help for commands · quit or Ctrl+C to stop.[/dim]",
            border_style="cyan",
        )
    )

    try:
        if rag is None:
            rag = _load_pipeline_with_status()
        elif hasattr(rag, "warm_up"):
            console.print("[yellow]🧠 Loading the answer model…[/yellow]")
            rag.warm_up()

        stats = rag.get_stats()
        if not stats.get("indexed", False):
            console.print("\n[yellow]⚠️  No index found — indexing documents…[/yellow]")
            index_documents(rag=rag)
            stats = rag.get_stats()

        console.print(_session_panel(rag, stats))
        _print_suggestions(rag)

        # Anything typed while the models were loading is discarded: keystrokes
        # from that phase arrive garbled in some terminals (observed as
        # UnicodeDecodeError on the first read) and the loading output mangles
        # their echo anyway. The first prompt starts from a clean buffer.
        _flush_typeahead()

        history: list[str] = []
        last_response: dict | None = None

        while True:
            try:
                question = _read_question()
                if not question:
                    continue
                if question.startswith("/") or question.lower() in ("quit", "exit", "q"):
                    if _handle_command(question, rag, history, last_response):
                        break
                    console.print()
                    continue

                # Static "thinking" line, then append-only streaming: no
                # cursor repainting at all, which renders correctly in every
                # terminal (animated spinners and Live repaints leave stale
                # lines in e.g. PyCharm's terminal).
                started = time.time()
                console.print(Text("🔍 Searching & thinking…", style="dim yellow"))
                streaming = [False]

                def _on_piece(piece: str, _streaming=streaming) -> None:
                    if not _streaming[0]:
                        _streaming[0] = True
                        console.print("\n[bold blue]🤖 ChatBot[/bold blue]")
                    print(piece, end="", flush=True)

                response = rag.ask(question, return_context=True, stream_callback=_on_piece)

                if streaming[0]:
                    print(flush=True)  # finish the streamed block
                else:
                    # Non-streaming engines (e.g. extractive) never call the
                    # callback — print the finished answer instead.
                    console.print("\n[bold blue]🤖 ChatBot[/bold blue]")
                    console.print(Text(response["answer"]))
                console.print()
                _print_answer(response, panel=False)
                _print_confidence(response)
                _print_timings(response, time.time() - started)
                history.append(question)
                last_response = response
                _flush_typeahead()
                console.print()
            except (KeyboardInterrupt, EOFError):
                # Ctrl+C, Ctrl+D, or a closed/piped stdin all end the session
                # cleanly instead of looping on the prompt forever.
                console.print("\n\n👋 [bold]Goodbye![/bold]")
                break
            except UnicodeDecodeError:
                # A multi-byte character (e.g. Persian) arrived mangled on
                # stdin. Drop the garbled bytes and just ask again.
                _flush_typeahead()
                console.print(
                    "[yellow]⚠️  Could not read that input — "
                    "please type your question again.[/yellow]\n"
                )
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                console.print()
                print_friendly_error(e)
                console.print()

    except FileNotFoundError as e:
        console.print(f"\n[red]❌ File not found: {e}[/red]")
        console.print(f"💡 Make sure you have documents in '{config.DOCS_DIR}'")
        raise SystemExit(1) from e
    except Exception as e:
        console.print()
        print_friendly_error(e)
        raise SystemExit(1) from e
