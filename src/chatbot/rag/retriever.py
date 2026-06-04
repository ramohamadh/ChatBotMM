"""
Retriever module.
Implements hybrid search combining semantic and keyword-based retrieval.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter
import re

from .chunker import normalize_persian

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining semantic and keyword search."""
    
    def __init__(
        self,
        vectorstore,
        embedding_generator,
        keyword_weight: float = 0.3,
        enable_hybrid: bool = True
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore: FAISSVectorStore instance
            embedding_generator: EmbeddingGenerator instance
            keyword_weight: Weight for keyword search (0.0 = semantic only, 1.0 = keyword only)
            enable_hybrid: Whether to enable hybrid search (if False, uses semantic only)
        """
        self.vectorstore = vectorstore
        self.embedding_generator = embedding_generator
        self.keyword_weight = keyword_weight
        self.enable_hybrid = enable_hybrid
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (chunk_dict, combined_score) tuples, sorted by score
        """
        # Normalize the query the same way indexed text was normalized.
        query = normalize_persian(query)

        # Semantic search
        query_embedding = self.embedding_generator.generate_embedding(query)
        semantic_results = self.vectorstore.search(query_embedding, top_k=top_k * 2)  # Get more for hybrid
        
        if not self.enable_hybrid:
            return semantic_results[:top_k]
        
        # Keyword search
        keyword_scores = self._keyword_search(query, semantic_results)
        
        # Combine scores
        combined_results = self._combine_scores(semantic_results, keyword_scores)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def _keyword_search(self, query: str, semantic_results: List[Tuple[Dict, float]]) -> Dict[int, float]:
        """
        Calculate keyword-based scores for semantic results.
        
        Args:
            query: Query string
            semantic_results: Results from semantic search
            
        Returns:
            Dictionary mapping result index to keyword score
        """
        # Extract keywords from query (simple tokenization)
        query_tokens = self._tokenize(query.lower())
        query_tf = Counter(query_tokens)
        
        keyword_scores = {}
        
        for idx, (chunk, _) in enumerate(semantic_results):
            chunk_tokens = self._tokenize(chunk["text"].lower())
            chunk_tf = Counter(chunk_tokens)
            
            # Calculate TF-IDF-like score (simplified)
            score = 0.0
            for token, query_count in query_tf.items():
                if token in chunk_tf:
                    # Simple term frequency matching
                    score += query_count * chunk_tf[token]
            
            # Normalize by chunk length
            if len(chunk_tokens) > 0:
                score = score / len(chunk_tokens)
            
            keyword_scores[idx] = score
        
        # Normalize scores to [0, 1]
        if keyword_scores:
            max_score = max(keyword_scores.values())
            if max_score > 0:
                keyword_scores = {k: v / max_score for k, v in keyword_scores.items()}
        
        return keyword_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (split on whitespace and punctuation).
        Supports Persian/Farsi text better.
        """
        # Normalize Persian/Arabic characters (shared with indexing path)
        text = normalize_persian(text)
        
        # Remove punctuation but keep Persian/Arabic characters
        # Keep alphanumeric, Persian/Arabic, and whitespace
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F]', ' ', text)
        
        tokens = text.split()
        # Filter out very short tokens (but keep Persian words which can be shorter)
        return [t for t in tokens if len(t) > 1]
    
    def _combine_scores(
        self,
        semantic_results: List[Tuple[Dict, float]],
        keyword_scores: Dict[int, float]
    ) -> List[Tuple[Dict, float]]:
        """
        Combine semantic and keyword scores.
        
        Args:
            semantic_results: Results from semantic search with similarity scores
            keyword_scores: Keyword scores indexed by result position
            
        Returns:
            List of (chunk_dict, combined_score) tuples
        """
        combined = []
        
        for idx, (chunk, semantic_score) in enumerate(semantic_results):
            keyword_score = keyword_scores.get(idx, 0.0)
            
            # Combine: (1 - keyword_weight) * semantic + keyword_weight * keyword
            combined_score = (
                (1 - self.keyword_weight) * semantic_score +
                self.keyword_weight * keyword_score
            )
            
            combined.append((chunk, combined_score))
        
        return combined

