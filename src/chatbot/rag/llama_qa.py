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
from contextlib import contextmanager, nullcontext

import llama_cpp
from llama_cpp import Llama

from .generative_qa import (
    NO_ANSWER_TEXT,
    REPETITION_PENALTY,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_context,
    build_user_prompt,
    is_looping,
    make_result,
)

logger = logging.getLogger(__name__)


@contextmanager
def _report_load_progress(callback: Callable[[float], None]):
    """Route llama.cpp's native model-load progress (0.0–1.0) to `callback`.

    The C API reports real load progress via llama_model_params
    .progress_callback, but the Python wrapper's Llama.__init__ doesn't expose
    it — it builds its params from llama_model_default_params(). Temporarily
    wrapping that factory injects the callback into the params every Llama
    created inside this context will use.
    """
    c_callback = llama_cpp.llama_progress_callback(
        # Return True to tell llama.cpp to keep loading.
        lambda progress, _user_data: callback(float(progress)) or True
    )
    original = llama_cpp.llama_model_default_params

    def with_progress():
        params = original()
        params.progress_callback = c_callback
        return params

    llama_cpp.llama_model_default_params = with_progress
    try:
        yield
    finally:
        llama_cpp.llama_model_default_params = original


class LlamaGenerativeQA:
    """Generative QA using a quantized GGUF model through llama.cpp."""

    def __init__(
        self,
        repo_id: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename: str = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        max_new_tokens: int = 300,
        max_context_chars: int = 3500,
        n_ctx: int = 8192,
        n_batch: int = 2048,
        temperature: float | None = None,
        load_progress: Callable[[float], None] | None = None,
    ):
        """
        Args:
            repo_id: HuggingFace repo holding the GGUF files.
            filename: GGUF file name (glob patterns allowed).
            max_new_tokens: Cap on generated answer length.
            max_context_chars: Cap on retrieved context passed to the model.
            n_ctx: Model context window in tokens.
            load_progress: Optional callback receiving model-load progress
                as a fraction (0.0–1.0), e.g. to drive a progress bar.
        """
        logger.info(f"Loading GGUF model: {repo_id} ({filename})")
        progress_scope = (
            _report_load_progress(load_progress) if load_progress else nullcontext()
        )
        # Downloads on first use, then loads from the local HF cache.
        with progress_scope:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                # Larger prefill batches are ~1.6x faster on this class of CPU
                # (measured: 26 tok/s at the default 512 vs 42 tok/s at 2048).
                n_batch=n_batch,
                # Offload every layer to the GPU when the build supports it
                # (Metal on Apple Silicon). Ignored by CPU-only builds.
                n_gpu_layers=-1,
                verbose=False,
            )
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.temperature = TEMPERATURE if temperature is None else temperature
        logger.info("GGUF model loaded successfully")

    def answer(
        self,
        question: str,
        chunks: list[dict],
        stream_callback: Callable[[str], None] | None = None,
        verbatim: bool = False,
    ) -> dict:
        """Generate an answer grounded in `chunks`; optionally stream pieces.

        verbatim=True disables the repetition penalty: quoting a JSON sample
        or a table from the context repeats tokens heavily, and the penalty
        makes the model refuse rather than copy. Used by the pipeline's
        false-refusal retry.
        """
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
            temperature=self.temperature,
            top_p=TOP_P,
            repeat_penalty=1.0 if verbatim else REPETITION_PENALTY,
        )

        if stream_callback is not None:
            pieces: list[str] = []
            text = ""
            for part in self.llm.create_chat_completion(stream=True, **kwargs):
                piece = part["choices"][0].get("delta", {}).get("content")
                if piece:
                    text += piece
                    # Stop the moment the model starts repeating itself —
                    # the repeated text is never streamed to the user, and
                    # cutting generation early also saves wall-clock time.
                    if is_looping(text):
                        logger.info("Generation loop detected — stopping early")
                        break
                    pieces.append(piece)
                    stream_callback(piece)
            answer = "".join(pieces).strip()
        else:
            out = self.llm.create_chat_completion(**kwargs)
            answer = (out["choices"][0]["message"]["content"] or "").strip()

        return make_result(answer, chunks)
