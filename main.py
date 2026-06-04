#!/usr/bin/env python
"""
Convenience entry point so you can run the CLI without installing the package:

    python main.py cli
    python main.py ask "..."
    python main.py index

It just makes the `src/` layout importable and delegates to chatbot.cli:main.
The installed `chatbot` console command does the same thing.
"""

import sys
from pathlib import Path

# Make src/ importable when running from a source checkout (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Ensure the CLI framework (typer) is installed BEFORE importing the CLI, so that
# `python main.py cli` can bootstrap itself instead of crashing on `import typer`.
# bootstrap.py has no third-party imports, so it is always importable.
from chatbot.bootstrap import ensure_cli_deps  # noqa: E402

ensure_cli_deps()

from chatbot.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
