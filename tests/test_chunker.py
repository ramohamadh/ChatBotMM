"""Tests for Persian normalization and text chunking (no heavy models needed)."""

from chatbot.rag.chunker import TextChunker, normalize_persian


def test_normalize_persian_arabic_to_persian_letters():
    assert normalize_persian("كتاب") == "کتاب"   # Arabic kaf -> Persian kaf
    assert normalize_persian("يك") == "یک"        # Arabic yeh -> Persian yeh


def test_normalize_persian_digits_and_diacritics():
    assert normalize_persian("صفحة ٣") == "صفحه 3"   # teh marbuta -> heh, Arabic digit -> latin
    # Diacritics (harakat) are stripped
    assert normalize_persian("مُحَمَّد") == "محمد"


def test_normalize_persian_handles_empty():
    assert normalize_persian("") == ""
    assert normalize_persian(None) is None


def test_chunker_splits_long_text_with_overlap():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "جمله اول. " * 50  # ~500 chars
    chunks = chunker.chunk_text(text, metadata={"filename": "t.txt"})
    assert len(chunks) > 1
    # Each chunk carries metadata and a chunk_id
    for c in chunks:
        assert "text" in c and c["text"].strip()
        assert c["metadata"]["filename"] == "t.txt"
        assert "chunk_id" in c["metadata"]


def test_chunker_normalizes_persian_in_cleaning():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_text("متن با حرف عربي و كاف")
    assert "ي" not in chunks[0]["text"]
    assert "ك" not in chunks[0]["text"]


def test_chunker_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_expand_identifier_question_rewrites_bare_id():
    from chatbot.rag.chunker import expand_identifier_question

    out = expand_identifier_question("ins چیست")
    assert "ins" in out
    assert "تعریف" in out  # explicit definition request
    # With the optional "فیلد" prefix and question mark too.
    out2 = expand_identifier_question("فیلد tinb چیست؟")
    assert "tinb" in out2 and "تعریف" in out2


def test_expand_identifier_question_leaves_normal_questions():
    from chatbot.rag.chunker import expand_identifier_question

    for q in (
        "موضوع صورتحساب چیست؟",          # Persian subject, not a Latin id
        "ins در کجا استفاده می‌شود؟",     # id but a different question shape
        "what is the ins field for?",     # English sentence, not the bare shape
    ):
        assert expand_identifier_question(q) == q
