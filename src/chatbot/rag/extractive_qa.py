"""
Question-Answering module.
Implements extractive QA using local transformer models.
"""

import logging

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class ExtractiveQA:
    """Extractive question-answering using local transformer models."""

    def __init__(self, model_name: str = "mrm8488/bert-multi-cased-finetuned-xquadv1"):
        """
        Initialize the QA model.

        Args:
            model_name: Name of the HuggingFace QA model to use
                       Default: multilingual BERT model fine-tuned for QA (supports Persian)
        """
        logger.info(f"Loading QA model: {model_name}")
        try:
            # Load model and tokenizer explicitly for better compatibility
            # with newer versions of transformers
            device = 0 if torch.cuda.is_available() else -1

            logger.info("Loading model and tokenizer...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

            # BERT models use standard tokenizers, no special handling needed
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=device
            )

            self.model_name = model_name
            logger.info("QA model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading QA model: {e}")
            raise

    def extract_answer(
        self,
        question: str,
        context: str,
        max_answer_length: int = 100,
        top_k: int = 1
    ) -> dict:
        """
        Extract answer from context given a question.

        Args:
            question: The question to answer
            context: The context text to search in
            max_answer_length: Maximum length of the answer
            top_k: Number of answers to return

        Returns:
            Dictionary with 'answer', 'score', 'start', 'end' keys
        """
        if not context.strip():
            return {
                "answer": "No context provided.",
                "score": 0.0,
                "start": 0,
                "end": 0
            }

        try:
            # Use the QA pipeline
            results = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=max_answer_length,
                top_k=top_k
            )

            # Handle both single result and list of results
            if isinstance(results, list):
                result = results[0] if results else {}
            else:
                result = results

            return {
                "answer": result.get("answer", "No answer found."),
                "score": result.get("score", 0.0),
                "start": result.get("start", 0),
                "end": result.get("end", 0)
            }

        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "score": 0.0,
                "start": 0,
                "end": 0
            }

    def extract_from_chunks(
        self,
        question: str,
        chunks: list[dict],
        max_context_length: int = 1024,
        max_answer_length: int = 200
    ) -> dict:
        """
        Extract answer from multiple chunks.
        Tries multiple strategies to get the best answer.

        Args:
            question: The question to answer
            chunks: List of chunk dictionaries with 'text' key
            max_context_length: Maximum context length to use
            max_answer_length: Maximum answer length

        Returns:
            Dictionary with answer, score, and source information
        """
        if not chunks:
            return {
                "answer": "No relevant chunks found.",
                "score": 0.0,
                "source_chunks": []
            }

        # Strategy 1: Try combined context from top chunks
        combined_context = self._combine_chunks(chunks, max_context_length)
        qa_result = self.extract_answer(question, combined_context, max_answer_length)

        # Strategy 2: If score is low, try individual chunks
        if qa_result["score"] < 0.15 or not qa_result.get("answer") or len(qa_result["answer"].strip()) < 5:
            logger.info(f"Low QA score ({qa_result['score']:.3f}), trying individual chunks...")

            best_score = qa_result["score"]
            best_answer = qa_result["answer"]

            # Try top 3 chunks individually
            for i, chunk in enumerate(chunks[:3]):
                chunk_text = chunk.get("text", "").strip()
                if len(chunk_text) < 50:  # Skip very short chunks
                    continue

                individual_result = self.extract_answer(question, chunk_text, max_answer_length)
                if individual_result["score"] > best_score:
                    best_score = individual_result["score"]
                    best_answer = individual_result["answer"]
                    logger.info(f"Found better answer in chunk {i+1} with score {best_score:.3f}")

            # Strategy 3: If still low, use best chunk as fallback
            if best_score < 0.1:
                logger.warning(f"Still low score ({best_score:.3f}), using best chunk as fallback")
                best_chunk = chunks[0].get("text", "").strip()

                # Extract meaningful portion (first few sentences or first paragraph)
                # Try to find sentences
                sentences = []
                for sep in ['. ', '.\n', '!', '?']:
                    parts = best_chunk.split(sep)
                    if len(parts) > 1:
                        sentences = parts[:3]
                        break

                if sentences:
                    fallback_answer = '. '.join(s.strip() for s in sentences if s.strip())[:600]
                else:
                    # Fallback to first 400 characters
                    fallback_answer = best_chunk[:400]

                if len(fallback_answer) > 600:
                    fallback_answer = fallback_answer[:600] + "..."

                best_answer = fallback_answer if fallback_answer else best_chunk[:400]
                best_score = 0.4  # Medium confidence for fallback

            qa_result = {
                "answer": best_answer,
                "score": best_score
            }

        # Add source information
        result = {
            "answer": qa_result["answer"],
            "score": qa_result["score"],
            "source_chunks": [
                {
                    "text": chunk.get("text", "")[:250] + "...",
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in chunks[:3]  # Include top 3 chunks
            ]
        }

        return result

    def _combine_chunks(self, chunks: list[dict], max_length: int) -> str:
        """
        Combine chunks into a single context string.
        Prioritizes top chunks and ensures meaningful context.

        Args:
            chunks: List of chunk dictionaries
            max_length: Maximum length of combined context

        Returns:
            Combined context string
        """
        combined = []
        current_length = 0

        # Use more chunks but prioritize top ones
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue

            chunk_len = len(chunk_text)

            # For top 3 chunks, try to include fully if possible
            if i < 3:
                if current_length + chunk_len <= max_length:
                    combined.append(chunk_text)
                    current_length += chunk_len + 2
                else:
                    # Try to fit at least 200 chars from top chunks
                    remaining = max_length - current_length
                    if remaining > 200:
                        combined.append(chunk_text[:remaining])
                        current_length += remaining
                    break
            else:
                # For other chunks, only add if there's significant space
                if current_length + chunk_len <= max_length:
                    combined.append(chunk_text)
                    current_length += chunk_len + 2
                else:
                    remaining = max_length - current_length
                    if remaining > 150:  # Only if meaningful amount remains
                        combined.append(chunk_text[:remaining])
                    break

        context = "\n\n".join(combined)
        logger.debug(f"Combined context length: {len(context)} chars from {len(combined)} chunks")
        return context

