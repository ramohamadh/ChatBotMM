"""
Command-line interface for ChatBotMM, built with Typer.

This is a thin shell over chatbot.commands. Heavy imports (transformers, torch,
faiss) are deferred until inside each command so that the `cli` bootstrap
command can install dependencies *before* they are needed.
"""

import logging
from pathlib import Path
from typing import Optional

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ChatBotMM — a fully local Persian/English RAG document chatbot.",
)


@app.command()
def index(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild (delete the existing index)."
    ),
) -> None:
    """Index the documents in data/docs into the vector store."""
    from . import commands

    commands.index_documents(force_reindex=force)


@app.command()
def rebuild() -> None:
    """Rebuild the index from scratch (alias for `index --force`)."""
    from . import commands

    commands.index_documents(force_reindex=True)
    logger.info("✅ Index rebuilt. Use 'chatbot ask' to ask questions.")


@app.command()
def ask(
    question: Optional[str] = typer.Argument(
        None, help="Question to ask. Omit to start interactive mode."
    ),
    context: bool = typer.Option(
        False, "--context", "-c", help="Include retrieved chunks in the output."
    ),
) -> None:
    """Ask a question (or start interactive mode if no question is given)."""
    from . import commands

    if question:
        response = commands.ask_single_question(question, return_context=context)
        commands._print_answer(response, show_context=context)
    else:
        commands.interactive_qa()


@app.command()
def cli(
    skip_install: bool = typer.Option(
        False, "--skip-install", help="Skip the dependency-install step."
    ),
) -> None:
    """
    One-shot bootstrap: install dependencies, build the index, then chat.

    Each step is skipped if it's already done, so it's safe to re-run.
    """
    print("🚀 ChatBotMM setup\n")

    # 1) Ensure dependencies are installed (may re-exec the process).
    if not skip_install:
        from .bootstrap import ensure_dependencies

        requirements = Path(__file__).resolve().parent.parent.parent / "requirements.txt"
        ensure_dependencies(requirements)

    # Import the heavy logic only after deps are guaranteed present.
    from . import commands

    # 2) Build the index if it doesn't exist yet.
    if commands.is_indexed():
        print("✅ Index already exists — skipping indexing.\n")
    else:
        print("📚 No index found — indexing documents...\n")
        commands.index_documents()
        print()

    # 3) Drop into interactive chat.
    commands.interactive_qa()


def main() -> None:
    """Entry point for the `chatbot` console script and `python -m chatbot`."""
    app()


if __name__ == "__main__":
    main()
