"""
Command-line interface for ChatBotMM, built with Typer.

This is a thin shell over chatbot.commands. Heavy imports (transformers, torch,
faiss) are deferred until inside each command so that the `cli` bootstrap
command can install dependencies *before* they are needed.
"""

import functools
import logging
from pathlib import Path

import typer

# WARNING by default so import-time INFO chatter (faiss, transformers, …)
# never reaches users; commands print their own friendly output instead.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="ChatBotMM — a fully local Persian/English RAG document chatbot.",
)


def _friendly_errors(fn):
    """Last-resort handler: turn any unhandled exception into a short
    human-readable message instead of a traceback."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Cancelled.")
            raise typer.Exit(code=130) from None
        except (typer.Exit, typer.Abort):
            raise
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - user-facing boundary
            try:
                # errors.py is dependency-light, but guard anyway: never let
                # the error reporter itself crash with a traceback.
                from .errors import print_friendly_error

                print_friendly_error(e)
            except Exception:  # noqa: BLE001
                print(f"❌ Error: {e}")
            raise typer.Exit(code=1) from None

    return wrapper


@app.callback(invoke_without_command=True)
@_friendly_errors
def _default(ctx: typer.Context) -> None:
    """Run `chatbot` with no command to chat interactively."""
    if ctx.invoked_subcommand is None:
        from . import commands

        commands.interactive_qa()


@app.command()
@_friendly_errors
def index(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild (delete the existing index)."
    ),
) -> None:
    """Index the documents in data/docs into the vector store."""
    from . import commands

    commands.index_documents(force_reindex=force)


@app.command()
@_friendly_errors
def rebuild() -> None:
    """Rebuild the index from scratch (alias for `index --force`)."""
    from . import commands

    commands.index_documents(force_reindex=True)
    commands.console.print("Use [bold]chatbot ask[/bold] to ask questions.")


@app.command()
@_friendly_errors
def ask(
    question: str | None = typer.Argument(
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
@_friendly_errors
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start the REST API server (docs at http://<host>:<port>/docs)."""
    import uvicorn

    # Pass the app object (not an import string) so this also works from a
    # source checkout via `python main.py serve`, where the package isn't
    # installed and only main.py's sys.path tweak makes `chatbot` importable.
    from .api import app as api_app

    print(f"🚀 Starting ChatBotMM API on http://{host}:{port}  (docs: /docs)")
    uvicorn.run(api_app, host=host, port=port)


@app.command()
@_friendly_errors
def cli(
    skip_install: bool = typer.Option(
        False, "--skip-install", help="Skip the dependency-install step."
    ),
) -> None:
    """
    One-shot bootstrap: install dependencies, build the index, then chat.

    Each step is skipped if it's already done, so it's safe to re-run.
    """
    print("🚀 ChatBot setup\n")

    # 1) Ensure dependencies are installed (may re-exec the process), plus the
    # optional fast backend (best-effort, never fails the bootstrap).
    if not skip_install:
        from .bootstrap import ensure_dependencies, ensure_optional_dependencies

        requirements = Path(__file__).resolve().parent.parent.parent / "requirements.txt"
        ensure_dependencies(requirements)
        ensure_optional_dependencies()

    # Import the heavy logic only after deps are guaranteed present.
    from . import commands

    # 2+3) interactive_qa handles the rest: staged model loading with status
    # output, indexing when no index exists, then the chat loop.
    commands.interactive_qa()


def main() -> None:
    """Entry point for the `chatbot` console script and `python -m chatbot`."""
    app()


if __name__ == "__main__":
    main()
