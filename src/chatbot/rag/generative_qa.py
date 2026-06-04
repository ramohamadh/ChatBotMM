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
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# System prompt instructs the model to answer ONLY from the provided context and
# to reply in the user's language (Persian). This keeps it grounded in the docs.
SYSTEM_PROMPT = (
    "تو یک دستیار پاسخگو هستی که فقط بر اساس متنِ زمینه (context) که در اختیارت "
    "قرار می‌گیرد پاسخ می‌دهی. پاسخ را به زبان فارسیِ روان و دقیق بنویس. "
    "اگر پاسخ در متن زمینه وجود نداشت، صادقانه بگو که در سند اطلاعاتی پیدا نشد و "
    "از خودت چیزی نساز."
)


class GenerativeQA:
    """Generative (abstractive) QA using a local instruction-tuned LLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 512,
    ):
        """
        Initialize the generative QA model.

        Args:
            model_name: HuggingFace causal-LM (instruction-tuned) model name.
            max_new_tokens: Max tokens to generate for an answer.
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
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        logger.info("Generative QA model loaded successfully")

    def _build_context(self, chunks: List[Dict], max_chars: int = 6000) -> str:
        """Join the retrieved chunks into a single context block (with sources)."""
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

    def answer(
        self,
        question: str,
        chunks: List[Dict],
    ) -> Dict:
        """
        Generate an answer to `question` grounded in the retrieved `chunks`.

        Returns a dict shaped like the extractive QA result so it's a drop-in
        replacement: {answer, score, source_chunks}.
        """
        if not chunks:
            return {
                "answer": "اطلاعاتی مرتبط با این سؤال در سند پیدا نشد.",
                "score": 0.0,
                "source_chunks": [],
            }

        context = self._build_context(chunks)

        user_prompt = (
            f"متن زمینه:\n{context}\n\n"
            f"سؤال: {question}\n\n"
            "بر اساس متن زمینهٔ بالا و فقط با تکیه بر آن، پاسخ بده."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic; grounded answers
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode the newly generated tokens, not the prompt.
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not answer:
            answer = "اطلاعاتی مرتبط با این سؤال در سند پیدا نشد."

        return {
            "answer": answer,
            "score": 1.0,  # generative model has no extractive confidence; report as N/A upstream
            "source_chunks": [
                {
                    "text": (chunk.get("text", "") or "")[:250] + "...",
                    "metadata": chunk.get("metadata", {}),
                }
                for chunk in chunks[:3]
            ],
        }
