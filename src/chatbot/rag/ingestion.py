"""
Document ingestion module.
Supports loading and extracting text from PDF, DOCX, TXT, and MD files.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import docx
import PyPDF2
import re

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles ingestion of various document formats."""
    
    def __init__(self, supported_extensions: Optional[set] = None):
        """
        Initialize the document ingester.
        
        Args:
            supported_extensions: Set of file extensions to support.
                                 Defaults to {'.pdf', '.docx', '.txt', '.md'}
        """
        self.supported_extensions = supported_extensions or {".pdf", ".docx", ".txt", ".md"}
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if a file format is supported."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def load_document(self, file_path: Path) -> List[Tuple[str, Dict]]:
        """
        Load a document and extract text with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of (text, metadata_dict) tuples. For PDFs this will contain one entry
            per page, while other formats will typically return a single entry.
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loading document: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_pdf(self, file_path: Path) -> List[Tuple[str, Dict]]:
        """Load text from a PDF file, returning one entry per page."""
        page_entries: List[Tuple[str, Dict]] = []
        
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as page_error:
                        logger.warning(f"Failed to extract text from {file_path} page {page_num + 1}: {page_error}")
                        page_text = ""
                    
                    page_text = page_text.strip()
                    if not page_text:
                        continue
                    
                    page_metadata = {
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": total_pages,
                        "source_id": f"{file_path.name}::page_{page_num + 1}"
                    }
                    
                    page_entries.append((page_text, page_metadata))
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
        
        if not page_entries:
            logger.warning(f"No text extracted from PDF: {file_path}")
        
        return page_entries
    
    def _load_docx(self, file_path: Path) -> List[Tuple[str, Dict]]:
        """Load text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
            
            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": "docx",
                "paragraph_count": len(paragraphs)
            }
            
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise
        
        if not full_text:
            logger.warning(f"No text extracted from DOCX: {file_path}")
            return []
        
        if not full_text.strip():
            logger.warning(f"No text extracted from file: {file_path}")
            return []
        
        return [(full_text, metadata)]
    
    def _load_text(self, file_path: Path) -> List[Tuple[str, Dict]]:
        """Load text from a TXT or MD file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                full_text = file.read()
            
            # Count lines and sections (markdown headers)
            lines = full_text.split("\n")
            section_count = len([line for line in lines if line.strip().startswith("#")])
            
            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "line_count": len(lines),
                "section_count": section_count if file_path.suffix == ".md" else 0
            }
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise
        
        return [(full_text, metadata)]
    
    def load_directory(self, directory: Path) -> List[Tuple[str, Dict]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of (text, metadata) tuples for each document
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents: List[Tuple[str, Dict]] = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and self.is_supported(file_path):
                try:
                    entries = self.load_document(file_path)
                    documents.extend(entries)
                    logger.info(f"Successfully loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} document sections from {directory}")
        return documents

