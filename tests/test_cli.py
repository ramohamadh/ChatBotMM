"""Tests for the Typer CLI and the dependency bootstrap logic."""

import pytest
from typer.testing import CliRunner

from chatbot import bootstrap
from chatbot.cli import app
from chatbot.rag import pipeline as pipeline_module

runner = CliRunner()


class _StubGen:
    def __init__(self, *a, **k):
        pass

    def answer(self, question, chunks):
        return {
            "answer": f"[stub] {question}",
            "score": 1.0,
            "source_chunks": [],
        }


@pytest.fixture
def stub_generative(monkeypatch):
    monkeypatch.setattr(pipeline_module, "GenerativeQA", _StubGen)


# --- CLI surface ---------------------------------------------------------


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "rebuild", "ask", "cli"):
        assert cmd in result.stdout


def test_cli_command_help_mentions_bootstrap():
    result = runner.invoke(app, ["cli", "--help"])
    assert result.exit_code == 0
    assert "--skip-install" in result.stdout


from chatbot import config  # noqa: E402

index_exists = (config.VECTORSTORE_DIR / "faiss_index.index").exists()


@pytest.mark.skipif(not index_exists, reason="no FAISS index in data/vectorstore")
def test_ask_command_end_to_end(stub_generative):
    result = runner.invoke(app, ["ask", "این سند درباره چیست؟"])
    assert result.exit_code == 0
    assert "[stub]" in result.stdout


# --- bootstrap -----------------------------------------------------------


def test_bootstrap_splits_cli_and_runtime_deps():
    # typer must be importable now (it's a CLI dep we depend on for these tests)
    assert bootstrap._missing(bootstrap._CLI_DEPS) == []
    # _REQUIRED is the union of both maps
    assert set(bootstrap._REQUIRED) == set(bootstrap._CLI_DEPS) | set(bootstrap._RUNTIME_DEPS)


def test_ensure_dependencies_is_noop_when_present(monkeypatch):
    # When nothing is missing, ensure_dependencies must not try to install.
    monkeypatch.setattr(bootstrap, "_missing", lambda deps: [])
    called = {"installed": False}

    def _fail(*a, **k):
        called["installed"] = True

    monkeypatch.setattr(bootstrap, "_install_and_reexec", _fail)
    bootstrap.ensure_dependencies()
    assert called["installed"] is False
