#!/usr/bin/env python
"""
QA benchmark for ChatBotMM — measure accuracy before/after tuning.

Runs the questions in eval_questions.json through the real pipeline and
scores the result against expected keywords.

Two modes:
  --retrieval-only   Fast (~seconds per question). Checks whether the expected
                     answer text is present in the CONTEXT handed to the
                     generator — measures retrieval + rerank + stitching
                     quality in isolation.
  (default)          Full. Generates the actual answer and scores it —
                     measures the whole pipeline. Slow on CPU (~1-2 min per
                     question).

Examples:
  python scripts/eval_qa.py --retrieval-only
  python scripts/eval_qa.py --retrieval-only --candidates 20 --max-length 512
  python scripts/eval_qa.py                  # full run (slow)

Compare the summary line across runs to decide a config change with data
instead of anecdotes.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatbot import config  # noqa: E402
from chatbot.rag.chunker import (  # noqa: E402
    expand_fragment,
    expand_identifier_question,
    formalize_question,
    normalize_persian,
)
from chatbot.rag.generative_qa import build_context  # noqa: E402

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def canon(text: str) -> str:
    """Normalize text so keyword matching survives spelling variation."""
    text = normalize_persian(text or "").translate(_DIGITS)
    text = text.replace("‌", " ")  # ZWNJ -> space
    return re.sub(r"\s+", " ", text).lower()


def matches(text: str, item: dict) -> bool:
    c = canon(text)
    any_kw = [canon(k) for k in item.get("expect_any", [])]
    all_kw = [canon(k) for k in item.get("expect_all", [])]
    ok = True
    if any_kw:
        ok = any(k in c for k in any_kw)
    if ok and all_kw:
        ok = all(k in c for k in all_kw)
    return ok


def build_pipeline(args):
    from chatbot.commands import get_default_rag_pipeline

    rag = get_default_rag_pipeline()
    if args.candidates:
        rag.rerank_candidates = args.candidates
    if args.max_length:
        # Recreate lazily with the new cap.
        rag._rerank_max_length = args.max_length
        rag._reranker = None
    if args.no_rerank:
        rag.rerank_enabled = False
    return rag


def retrieval_context(rag, question: str) -> tuple[str, float]:
    """Reproduce the retrieval side of RAGPipeline.ask() and return the
    context string the generator would see, plus elapsed seconds."""
    q = expand_fragment(expand_identifier_question(formalize_question(question)))
    started = time.time()
    reranker = rag.reranker
    if reranker is not None:
        candidates = rag.retriever.retrieve(q, top_k=rag.rerank_candidates)
        retrieved = reranker.rerank(q, candidates, top_k=rag.rerank_top_k)
    else:
        retrieved = rag.retriever.retrieve(q, top_k=rag.top_k)
    retrieved = rag._with_neighbors(retrieved)
    if rag._is_overview_question(q):
        intro = rag._intro_chunks()
        intro_texts = {c["text"] for c in intro}
        retrieved = [(c, 1.0) for c in intro] + [
            (c, s) for c, s in retrieved if c["text"] not in intro_texts
        ]
    elapsed = time.time() - started
    chunks = [c for c, _ in retrieved]
    return build_context(chunks, config.GENERATIVE_MAX_CONTEXT_CHARS), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "eval_questions.json"))
    parser.add_argument("--candidates", type=int, help="override RERANK_CANDIDATES")
    parser.add_argument("--max-length", type=int, help="override RERANK_MAX_LENGTH")
    parser.add_argument("--no-rerank", action="store_true", help="disable the reranker")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument(
        "--temperature", type=float, help="override generation temperature (full mode)"
    )
    args = parser.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    print("Loading pipeline...")
    rag = build_pipeline(args)
    if not rag.is_indexed:
        print("No index found — run `chatbot index` first.")
        return 1
    if not args.retrieval_only:
        rag.warm_up()
        if args.temperature is not None and hasattr(rag.qa, "temperature"):
            rag.qa.temperature = args.temperature
            print(f"generation temperature override: {args.temperature}")

    mode = "retrieval-only" if args.retrieval_only else "full"
    rr = "off" if args.no_rerank else (
        f"{rag.rerank_candidates} cand @ {rag._rerank_max_length} tok"
    )
    print(f"Mode: {mode} | rerank: {rr} | questions: {len(items)}\n")

    hits, times = 0, []
    for i, item in enumerate(items, 1):
        q = item["question"]
        if args.retrieval_only:
            text, elapsed = retrieval_context(rag, q)
        else:
            started = time.time()
            resp = rag.ask(q)
            elapsed = time.time() - started
            text = resp["answer"]
        ok = matches(text, item)
        hits += ok
        times.append(elapsed)
        mark = "PASS" if ok else "FAIL"
        print(f"[{i:2}/{len(items)}] {mark}  {elapsed:6.1f}s  {q}")
        if not ok and not args.retrieval_only:
            print(f"        answer: {text[:120]!r}")

    total = len(items)
    print(
        f"\n=== {mode} | rerank {rr} ==="
        f"\nhit rate : {hits}/{total} ({100 * hits / total:.0f}%)"
        f"\navg time : {sum(times) / total:.1f}s | max: {max(times):.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
