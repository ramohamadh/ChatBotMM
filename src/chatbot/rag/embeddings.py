"""
Embedding generation module.
Uses sentence-transformers for local, offline embedding generation.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers models."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
                       Default: multilingual model supporting 50+ languages including Persian
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        # E5-family models are trained with role prefixes; retrieval quality
        # degrades noticeably without them.
        self._use_e5_prefixes = "e5" in model_name.lower()
        # sentence-transformers >= 5.6 renamed get_sentence_embedding_dimension.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.embedding_dimension = get_dim()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dimension}")

    def generate_embeddings(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dimension)
        """
        if not texts:
            return np.array([])

        if self._use_e5_prefixes:
            texts = [f"passage: {text}" for text in texts]

        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        logger.info(f"Generated embeddings of shape {embeddings.shape}")
        return embeddings

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dimension,)
        """
        if self._use_e5_prefixes:
            text = f"query: {text}"
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dimension

