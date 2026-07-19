"""
Quantized answer engine backed by llama.cpp.

Runs the same Qwen instruct model as generative_qa.py but as a 4-bit GGUF via
llama-cpp-python — typically 2-4x faster on CPU and ~4x less RAM than float32
transformers, with near-identical answer quality. Drop-in replacement for
GenerativeQA: same answer() signature, same result shape, same streaming
callback support.
"""

import logging
from collections.abc import Callable

from llama_cpp import Llama

from .generative_qa import (
    NO_ANSWER_TEXT,
    REPETITION_PENALTY,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_context,
    build_user_prompt,
    make_result,
)

logger = logging.getLogger(__name__)


class LlamaGenerativeQA:
    """Generative QA using a quantized GGUF model through llama.cpp."""

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename: str = "*q4_k_m.gguf",
        max_new_tokens: int = 300,
        max_context_chars: int = 3500,
        n_ctx: int = 4096,
        n_batch: int = 2048,
    ):
        """
        Args:
            repo_id: HuggingFace repo holding the GGUF files.
            filename: GGUF file name (glob patterns allowed).
            max_new_tokens: Cap on generated answer length.
            max_context_chars: Cap on retrieved context passed to the model.
            n_ctx: Model context window in tokens.
        """
        logger.info(f"Loading GGUF model: {repo_id} ({filename})")
        # Downloads on first use, then loads from the local HF cache.
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            # Larger prefill batches are ~1.6x faster on this class of CPU
            # (measured: 26 tok/s at the default 512 vs 42 tok/s at 2048).
            n_batch=n_batch,
            verbose=False,
        )
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        logger.info("GGUF model loaded successfully")

    def answer(
        self,
        question: str,
        chunks: list[dict],
        stream_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """Generate an answer grounded in `chunks`; optionally stream pieces."""
        if not chunks:
            return {"answer": NO_ANSWER_TEXT, "score": 0.0, "source_chunks": []}

        context = build_context(chunks, self.max_context_chars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context)},
        ]
        kwargs = dict(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPETITION_PENALTY,
        )

        if stream_callback is not None:
            pieces: list[str] = []
            for part in self.llm.create_chat_completion(stream=True, **kwargs):
                piece = part["choices"][0].get("delta", {}).get("content")
                if piece:
                    pieces.append(piece)
                    stream_callback(piece)
            answer = "".join(pieces).strip()
        else:
            out = self.llm.create_chat_completion(**kwargs)
            answer = (out["choices"][0]["message"]["content"] or "").strip()

        return make_result(answer, chunks)
