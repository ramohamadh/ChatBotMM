"""
Bootstrap helpers: ensure runtime dependencies are installed.

Auto-installing into the *currently running* interpreter is tricky — packages
imported at startup won't pick up newly installed ones mid-process. The robust
pattern (used here) is: detect what's missing, pip-install it into the *same*
interpreter, and if anything was actually installed, re-exec the process so the
fresh packages import cleanly.
"""

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# typer is needed just to build the CLI itself, so it must be present before we
# can even import chatbot.cli. Kept separate from the heavy ML runtime deps.
_CLI_DEPS = {
    "typer": "typer",
}

# The heavy pipeline dependencies, needed before indexing / answering.
_RUNTIME_DEPS = {
    "numpy": "numpy",
    "faiss-cpu": "faiss",
    "pypdf": "pypdf",
    "pdfplumber": "pdfplumber",
    "python-docx": "docx",
    "sentence-transformers": "sentence_transformers",
    "transformers": "transformers",
    "torch": "torch",
    "sentencepiece": "sentencepiece",
}

# Everything, for the full `cli` bootstrap.
_REQUIRED = {**_CLI_DEPS, **_RUNTIME_DEPS}

# Optional speed-ups: installed best-effort by `cli`, never required. The app
# falls back to the transformers backend when llama-cpp-python is missing.
_OPTIONAL_DEPS = {
    "llama-cpp-python": "llama_cpp",
}
# Prebuilt CPU wheels (avoids compiling from source where a wheel exists).
_LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# Guard env var so a re-exec'd child never tries to install again (prevents loops).
_REEXEC_FLAG = "CHATBOT_BOOTSTRAPPED"


def _missing(deps: dict[str, str]) -> list[str]:
    """Return the pip package names from `deps` whose import is missing."""
    return [pkg for pkg, imp in deps.items() if importlib.util.find_spec(imp) is None]


def _install_and_reexec(
    missing_packages: list[str], requirements_file: Path | None = None
) -> None:
    """pip-install the given packages (or a requirements file), then re-exec."""
    # Avoid infinite re-exec loops if installation didn't actually fix things.
    if os.environ.get(_REEXEC_FLAG):
        raise RuntimeError(
            "Dependencies are still missing after install: "
            f"{', '.join(missing_packages)}. Install them manually with: "
            f"pip install {' '.join(missing_packages)}"
        )

    print("📦 Installing dependencies (first-time setup)...")
    print("   This can take several minutes on the first install — please wait.")
    # -q keeps the output clean (no "Requirement already satisfied" walls).
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if requirements_file and Path(requirements_file).exists():
        cmd += ["-r", str(requirements_file)]
        print(f"   using {requirements_file}")
    else:
        cmd += missing_packages

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Dependency installation failed. Check your internet connection and "
            f"try manually: {' '.join(cmd)}"
        ) from e

    print("✅ Dependencies installed. Restarting...\n")
    # Re-exec the exact same command so newly installed packages import cleanly.
    os.environ[_REEXEC_FLAG] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def ensure_cli_deps() -> None:
    """
    Ensure the deps needed just to *build the CLI* (typer) are present.

    Called from main.py before importing chatbot.cli, so that `python main.py cli`
    can self-install typer instead of crashing on `import typer`.
    """
    missing = _missing(_CLI_DEPS)
    if missing:
        _install_and_reexec(missing)


def ensure_dependencies(requirements_file: Path | None = None) -> None:
    """
    Make sure all runtime dependencies are installed.

    If all imports are present, this is a no-op. Otherwise it pip-installs the
    requirements file (preferred) or the missing packages, then re-execs the
    process so the freshly installed modules can be imported.
    """
    missing = _missing(_REQUIRED)
    if not missing:
        logger.debug("All runtime dependencies already present.")
        return
    _install_and_reexec(missing, requirements_file)


def ensure_optional_dependencies() -> None:
    """
    Best-effort install of the fast llama.cpp answer backend.

    Unlike ensure_dependencies this NEVER fails the bootstrap: on machines
    where no wheel matches and no compiler is available, the app simply keeps
    using the transformers backend. No re-exec is needed — the package is
    imported lazily, long after this point.
    """
    missing = _missing(_OPTIONAL_DEPS)
    if not missing:
        return

    print("📦 Installing the fast answer backend (llama-cpp-python)...")
    print("   This can take several minutes if it compiles from source — please wait.")
    cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        "--extra-index-url", _LLAMA_CPP_WHEEL_INDEX,
        *missing,
    ]
    for attempt in (1, 2):  # one retry for flaky downloads
        try:
            subprocess.run(cmd, check=True)
            importlib.invalidate_caches()
            print("✅ Fast backend installed.")
            return
        except subprocess.CalledProcessError:
            if attempt == 1:
                print("   Retrying once...")
    print(
        "⚠️  Could not install llama-cpp-python — continuing with the standard "
        "(slower) backend. You can retry manually with: "
        f"pip install llama-cpp-python --extra-index-url {_LLAMA_CPP_WHEEL_INDEX}"
    )
