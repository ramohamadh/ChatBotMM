"""Tests for GenerativeQA context-building (without downloading the LLM)."""

from chatbot.rag.generative_qa import SYSTEM_PROMPT, GenerativeQA


class _StubGen(GenerativeQA):
    """Subclass that skips the expensive model load."""

    def __init__(self):
        self.max_new_tokens = 64
        self.max_context_chars = 3500
        self.device = "cpu"


def test_build_context_adds_source_and_page_headers():
    g = _StubGen()
    chunks = [
        {"text": "متن یک", "metadata": {"page_number": 12}},
        {"text": "متن دو", "metadata": {"page_number": 36}},
    ]
    ctx = g._build_context(chunks)
    assert "[منبع 1 - صفحه 12]" in ctx
    assert "[منبع 2 - صفحه 36]" in ctx
    assert "متن یک" in ctx and "متن دو" in ctx


def test_build_context_without_page_number():
    g = _StubGen()
    ctx = g._build_context([{"text": "بدون شماره صفحه", "metadata": {}}])
    assert "[منبع 1]" in ctx


def test_build_context_respects_max_chars():
    g = _StubGen()
    big = {"text": "x" * 10000, "metadata": {}}
    ctx = g._build_context([big], max_chars=500)
    assert len(ctx) <= 600  # header + truncated body


def test_system_prompt_is_persian():
    # Contains Persian text instructing Persian answers
    assert "فارسی" in SYSTEM_PROMPT
