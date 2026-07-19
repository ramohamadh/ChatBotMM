"""
Text chunking module.
Implements smart chunking with overlap and metadata preservation.
"""

import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


# Arabic -> Persian character normalization plus removal of diacritics and the
# zero-width-non-joiner inconsistencies that make Persian text fail to match.
_PERSIAN_CHAR_MAP = {
    'ي': 'ی',   # Arabic yeh -> Persian yeh
    'ك': 'ک',   # Arabic kaf -> Persian kaf
    'ى': 'ی',   # alef maksura -> yeh
    'ۀ': 'ه',
    'ة': 'ه',
    'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
    'ؤ': 'و',
    'ئ': 'ی',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}
# Arabic diacritics (harakat) and tatweel — strip them entirely.
_PERSIAN_DIACRITICS = re.compile(r'[ً-ٰٟـ]')


def normalize_persian(text: str) -> str:
    """Normalize Persian/Arabic text so that equivalent forms match.

    Used both when indexing documents and when handling a user query, so the
    two are always normalized the same way.
    """
    if not text:
        return text
    for src, dst in _PERSIAN_CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _PERSIAN_DIACRITICS.sub('', text)
    return text


class TextChunker:
    """Handles text chunking with overlap and metadata."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks with overlap and metadata.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata dictionary to attach to each chunk
            
        Returns:
            List of chunk dictionaries, each containing:
            - text: The chunk text
            - metadata: Original metadata plus chunk-specific info
            - chunk_id: Unique identifier for the chunk
        """
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start, end - int(self.chunk_size * 0.2))
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create chunk metadata
                chunk_metadata = (metadata.copy() if metadata else {})
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_length": len(chunk_text)
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving paragraph breaks."""
        text = normalize_persian(text)
        text = text.replace('\r\n', '\n')
        text = re.sub(r'\t+', ' ', text)
        # Collapse multiple spaces but keep single spaces
        text = re.sub(r' {2,}', ' ', text)
        # Normalize newline groups to at most two
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the best sentence boundary within the given range.
        
        Args:
            text: The full text
            start: Start position to search from
            end: End position to search to
            
        Returns:
            Position of sentence boundary, or end if not found
        """
        # Look for sentence endings: . ! ? ؟ (Persian) followed by space or newline
        pattern = r'[.!?؟…]\s+'
        matches = list(re.finditer(pattern, text[start:end]))
        
        if matches:
            # Use the last match before the end
            last_match = matches[-1]
            return start + last_match.end()
        
        return end
    
    def chunk_documents(self, documents: List[tuple]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of (text, metadata) tuples
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for text, metadata in documents:
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks

