"""
User-facing error reporting.

Translates the exceptions this app realistically hits (no internet during the
model download, out of RAM, missing dependency, missing files) into short
plain-language messages with a hint — instead of a Python traceback that makes
users think the program is broken.

This module must stay light: no torch/transformers imports, so it is safe to
import even when the heavy dependencies are missing (which is itself one of
the errors it explains).
"""

import logging
from collections.abc import Iterator

from rich.console import Console

from . import config

logger = logging.getLogger(__name__)
console = Console()

# Exception class names (from requests/httpx/urllib3/huggingface_hub) that mean
# "the network or the Hub is unreachable" — matched by name so we don't have to
# import those libraries here.
_NETWORK_ERROR_NAMES = {
    "ConnectionError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "Timeout",
    "TimeoutError",
    "ProxyError",
    "SSLError",
    "NameResolutionError",
    "MaxRetryError",
    "HfHubHTTPError",
    "LocalEntryNotFoundError",
    "OfflineModeIsEnabled",
}


def _error_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield `error` and every exception it was raised from."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_network_error(error: BaseException) -> bool:
    for err in _error_chain(error):
        if isinstance(err, (ConnectionError, TimeoutError)):
            return True
        if type(err).__name__ in _NETWORK_ERROR_NAMES:
            return True
        text = str(err).lower()
        if (
            "name resolution" in text
            or "network is unreachable" in text
            or "connection refused" in text
            or "connection aborted" in text
            or "connection error" in text
            or "failed to establish" in text
            # huggingface_hub's retry loop swallows the original ConnectError
            # and raises a bare RuntimeError with this message instead.
            or "cannot send a request" in text
            or "huggingface.co" in text
        ):
            return True
    return False


def _is_memory_error(error: BaseException) -> bool:
    for err in _error_chain(error):
        if isinstance(err, MemoryError):
            return True
        text = str(err).lower()
        if "out of memory" in text or "cannot allocate memory" in text:
            return True
    return False


def print_friendly_error(error: BaseException) -> None:
    """Explain `error` in plain language on the console, without a traceback.

    The full traceback is still recorded at DEBUG level for diagnostics.
    """
    logger.debug("Full traceback:", exc_info=error)

    if _is_memory_error(error):
        console.print("[red]❌ Not enough memory (RAM) to run the model.[/red]")
        console.print(
            "💡 Close other programs, or use the smaller model: set "
            "GENERATIVE_MODEL in src/chatbot/config.py to "
            "'Qwen/Qwen2.5-0.5B-Instruct'."
        )
    elif _is_network_error(error):
        console.print("[red]❌ Could not reach huggingface.co to download the models.[/red]")
        console.print(
            "💡 Check your internet connection (or proxy/VPN) and try again — "
            "after the first successful download everything runs offline."
        )
    elif isinstance(error, (ImportError, ModuleNotFoundError)):
        console.print(f"[red]❌ A required package is missing: {error}[/red]")
        console.print(
            "💡 Run 'python main.py cli' to install everything automatically, "
            "or 'pip install -r requirements.txt'."
        )
    elif isinstance(error, FileNotFoundError):
        console.print(f"[red]❌ File not found: {error}[/red]")
        console.print(f"💡 Put your documents (PDF, DOCX, TXT or MD) in '{config.DOCS_DIR}'.")
    else:
        console.print(f"[red]❌ Unexpected error: {error}[/red]")
        console.print(
            "💡 If this keeps happening, try rebuilding the index with "
            "'chatbot rebuild' and check that the files in "
            f"'{config.DOCS_DIR}' open correctly."
        )
