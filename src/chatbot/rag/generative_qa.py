"""
Generative question-answering module.

Unlike the extractive QA model (which can only copy a span of words out of the
context), this uses a small local instruction-tuned LLM to *generate* a fluent
answer in the same language as the question. This is what makes the bot able to
actually answer in natural Persian instead of returning chopped-up fragments.

Default model: Qwen2.5-1.5B-Instruct — strong multilingual support (good Persian),
runs fully offline on CPU or Apple-Silicon MPS, and fits comfortably in 8 GB RAM.
"""

import logging
from collections.abc import Callable
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)


# System prompt instructs the model to answer ONLY from the provided context and
# to reply in the user's language (Persian). This keeps it grounded in the docs.
# Kept short and non-self-referential on purpose: when a small model cannot
# find an answer it tends to parrot its own instructions back, so the less
# quotable text here, the better.
SYSTEM_PROMPT = (
    "فقط بر اساس متن زمینه پاسخ بده و به فارسی روان بنویس. "
    "پاسخ کامل و با جزئیات باشد: همهٔ نکته‌های مرتبط با سؤال را از متن زمینه بیاور "
    "و اگر چند مورد مرتبط در متن هست، همهٔ آن‌ها را فهرست کن. "
    "هیچ جمله‌ای را تکرار نکن و چیزی از خودت نساز. "
    "اگر پاسخ در متن زمینه نبود، بنویس: در سند اطلاعاتی پیدا نشد."
)

NO_ANSWER_TEXT = "اطلاعاتی مرتبط با این سؤال در سند پیدا نشد."

# Decoding settings shared by both answer backends (transformers and
# llama.cpp): light sampling breaks repetition loops without the word-mutation
# failure mode of greedy + tight n-gram blocking.
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.15


def build_context(chunks: list[dict], max_chars: int) -> str:
    """Join retrieved chunks into a single context block with source headers."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        page = chunk.get("metadata", {}).get("page_number")
        header = f"[منبع {i} - صفحه {page}]" if page else f"[منبع {i}]"
        block = f"{header}\n{text}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_user_prompt(question: str, context: str) -> str:
    # No closing instruction line: small models tend to echo it verbatim at
    # the start of their answer. The system prompt already carries the rules.
    return f"متن زمینه:\n{context}\n\nسؤال: {question}"


def make_result(answer: str, chunks: list[dict]) -> dict:
    """Shape the QA result dict shared by both backends."""
    return {
        "answer": answer or NO_ANSWER_TEXT,
        "score": 1.0,
        "source_chunks": [
            {
                "text": (chunk.get("text", "") or "")[:250] + "...",
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks[:3]
        ],
    }


class GenerativeQA:
    """Generative (abstractive) QA using a local instruction-tuned LLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 300,
        max_context_chars: int = 3500,
    ):
        """
        Initialize the generative QA model.

        Args:
            model_name: HuggingFace causal-LM (instruction-tuned) model name.
            max_new_tokens: Max tokens to generate for an answer.
            max_context_chars: Cap on the retrieved context passed to the model
                (a longer context means noticeably slower answers on CPU).
        """
        logger.info(f"Loading generative QA model: {model_name}")

        # Pick the best available device. MPS = Apple Silicon GPU.
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        logger.info(f"Using device: {self.device} (dtype={dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        logger.info("Generative QA model loaded successfully")

    def _build_context(self, chunks: list[dict], max_chars: int | None = None) -> str:
        """Join the retrieved chunks into a single context block (with sources)."""
        return build_context(chunks, max_chars if max_chars is not None else self.max_context_chars)

    def answer(
        self,
        question: str,
        chunks: list[dict],
        stream_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Generate an answer to `question` grounded in the retrieved `chunks`.

        Args:
            stream_callback: If given, called with each piece of text as it is
                generated, so the caller can display the answer live instead
                of waiting for the whole generation to finish.

        Returns a dict shaped like the extractive QA result so it's a drop-in
        replacement: {answer, score, source_chunks}.
        """
        if not chunks:
            return {
                "answer": NO_ANSWER_TEXT,
                "score": 0.0,
                "source_chunks": [],
            }

        context = self._build_context(chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context)},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            # Light sampling instead of greedy: greedy decoding on a small
            # model tends to loop on one phrase, and blocking repeats with
            # no_repeat_ngram forces it to mutate words into nonsense instead.
            # A low temperature breaks loops naturally while staying grounded.
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            # Loose safety net against sentence-level loops. Safe combined
            # with sampling (the word-mutation failure mode only appears with
            # greedy decoding + a tight n-gram limit).
            no_repeat_ngram_size=10,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if stream_callback is not None:
            # Stream: generate in a worker thread while this thread hands each
            # decoded piece to the callback as soon as it is ready.
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs["streamer"] = streamer

            def _generate() -> None:
                with torch.no_grad():
                    self.model.generate(**generate_kwargs)

            worker = Thread(target=_generate, daemon=True)
            worker.start()
            pieces: list[str] = []
            for piece in streamer:
                if piece:
                    pieces.append(piece)
                    stream_callback(piece)
            worker.join()
            answer = "".join(pieces).strip()
        else:
            with torch.no_grad():
                generated = self.model.generate(**generate_kwargs)

            # Only decode the newly generated tokens, not the prompt.
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return make_result(answer, chunks)
