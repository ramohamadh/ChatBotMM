"""
Vectorstore module.
Manages FAISS index for storing and retrieving document embeddings.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manages FAISS index for vector similarity search."""
    
    def __init__(self, embedding_dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
        """
        self.embedding_dimension = embedding_dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []  # Store chunks with metadata
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize a new FAISS index."""
        # Use Inner Product (IP) index since embeddings are normalized
        # IP is equivalent to cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings and chunks to the index.
        
        Args:
            embeddings: numpy array of shape (n_chunks, embedding_dimension)
            chunks: List of chunk dictionaries with text and metadata
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings but {len(chunks)} chunks")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector of shape (embedding_dimension,)
            query_embedding: numpy array of shape (embedding_dimension,)
            top_k: Number of results to return
            
        Returns:
            List of (chunk_dict, similarity_score) tuples, sorted by similarity
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Reshape query embedding for FAISS
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Get results
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            # FAISS pads missing results with -1; a negative index would silently
            # pick the wrong chunk via Python negative indexing.
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    def save(self, directory: Path, index_name: str = "faiss_index", metadata_name: str = "metadata.json"):
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the index and metadata
            index_name: Name for the FAISS index file
            metadata_name: Name for the metadata JSON file
        """
        directory.mkdir(parents=True, exist_ok=True)
        
        index_path = directory / f"{index_name}.index"
        metadata_path = directory / metadata_name
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save chunks metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, directory: Path, index_name: str = "faiss_index", metadata_name: str = "metadata.json"):
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory containing the saved index and metadata
            index_name: Name of the FAISS index file
            metadata_name: Name of the metadata JSON file
        """
        index_path = directory / f"{index_name}.index"
        metadata_path = directory / metadata_name
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load chunks metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks from metadata")
        
        # Verify dimension
        if self.index.d != self.embedding_dimension:
            logger.warning(
                f"Dimension mismatch: index has {self.index.d}, "
                f"expected {self.embedding_dimension}"
            )
            self.embedding_dimension = self.index.d
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dimension
        }
    
    def clear(self):
        """Clear all data from the vector store."""
        self._initialize_index()
        self.chunks = []
        logger.info("Vector store cleared")

