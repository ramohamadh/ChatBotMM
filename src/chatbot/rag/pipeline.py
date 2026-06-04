"""
RAG Pipeline module.
High-level interface for the complete RAG system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .ingestion import DocumentIngester
from .chunker import TextChunker
from .embeddings import EmbeddingGenerator
from .vectorstore import FAISSVectorStore
from .retriever import HybridRetriever
from .extractive_qa import ExtractiveQA
from .generative_qa import GenerativeQA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    High-level RAG pipeline managing all components.
    
    Usage:
        rag = RAGPipeline("docs/")
        rag.index_documents()
        answer = rag.ask("What is the main topic?")
    """
    
    def __init__(
        self,
        docs_directory: str = "docs",
        vectorstore_directory: str = "vectorstore",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        qa_model: str = "mrm8488/bert-multi-cased-finetuned-xquadv1",
        top_k: int = 5,
        hybrid_search: bool = True,
        keyword_weight: float = 0.3,
        use_generative: bool = True,
        generative_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            docs_directory: Path to directory containing documents
            vectorstore_directory: Path to directory for storing vector index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of sentence-transformers model (default: multilingual)
            qa_model: Name of HuggingFace QA model (default: multilingual)
            top_k: Number of chunks to retrieve
            hybrid_search: Whether to enable hybrid search
            keyword_weight: Weight for keyword search in hybrid mode
        """
        self.docs_directory = Path(docs_directory)
        self.vectorstore_directory = Path(vectorstore_directory)
        self.vectorstore_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        self.ingester = DocumentIngester()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        self.vectorstore = FAISSVectorStore(
            embedding_dimension=self.embedding_generator.dimension
        )
        
        self.retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            embedding_generator=self.embedding_generator,
            keyword_weight=keyword_weight,
            enable_hybrid=hybrid_search
        )
        
        # Answer engine: generative (fluent, chat-like) is the default.
        # The extractive model is kept as a lightweight fallback.
        self.use_generative = use_generative
        if use_generative:
            self.qa = GenerativeQA(model_name=generative_model)
        else:
            self.qa = ExtractiveQA(model_name=qa_model)

        self.top_k = top_k
        self.is_indexed = False
        
        # Check if index already exists and load it
        index_path = self.vectorstore_directory / "faiss_index.index"
        if index_path.exists():
            try:
                self.vectorstore.load(self.vectorstore_directory)
                self.is_indexed = True
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
        
        logger.info("RAG pipeline initialized")
    
    def index_documents(self, force_reindex: bool = False) -> Dict:
        """
        Index all documents in the docs directory.
        
        Args:
            force_reindex: If True, reindex even if index exists
            
        Returns:
            Dictionary with indexing statistics
        """
        # Check if index already exists
        index_path = self.vectorstore_directory / "faiss_index.index"
        if index_path.exists() and not force_reindex:
            logger.info("Loading existing index...")
            try:
                self.vectorstore.load(self.vectorstore_directory)
                self.is_indexed = True
                stats = self.vectorstore.get_stats()
                logger.info(f"Loaded existing index with {stats['total_chunks']} chunks")
                return stats
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}. Reindexing...")
        
        # Load documents
        logger.info(f"Loading documents from {self.docs_directory}")
        if not self.docs_directory.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_directory}")
        
        documents = self.ingester.load_directory(self.docs_directory)
        
        if not documents:
            logger.warning("No documents found to index")
            return {"total_chunks": 0, "total_documents": 0}
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        if not chunks:
            logger.warning("No chunks created from documents")
            return {"total_chunks": 0, "total_documents": len(documents)}
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts, show_progress=True)
        
        # Add to vectorstore
        logger.info("Adding to vectorstore...")
        self.vectorstore.add_embeddings(embeddings, chunks)
        
        # Save vectorstore
        logger.info("Saving vectorstore...")
        self.vectorstore.save(self.vectorstore_directory)
        
        self.is_indexed = True
        
        stats = self.vectorstore.get_stats()
        unique_docs = {metadata.get("filename") for _, metadata in documents if metadata.get("filename")}
        stats["total_documents"] = len(unique_docs) if unique_docs else len(documents)
        logger.info(f"Indexing complete: {stats['total_chunks']} chunks from {len(documents)} documents")
        
        return stats
    
    def ask(self, question: str, return_context: bool = False) -> Dict:
        """
        Ask a question and get an answer from the indexed documents.
        
        Args:
            question: The question to ask
            return_context: If True, include retrieved chunks in response
            
        Returns:
            Dictionary with:
            - answer: The extracted answer
            - score: Confidence score
            - retrieved_chunks: List of retrieved chunks (if return_context=True)
            - source_chunks: Source information for the answer
        """
        if not self.is_indexed:
            raise ValueError("Documents must be indexed first. Call index_documents()")
        
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant chunks
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        
        if not retrieved:
            return {
                "answer": "No relevant information found in the documents.",
                "score": 0.0,
                "retrieved_chunks": [],
                "source_chunks": []
            }
        
        # Log retrieved chunks for debugging
        logger.info(f"Retrieved {len(retrieved)} chunks:")
        for i, (chunk, score) in enumerate(retrieved[:3], 1):
            logger.debug(f"  {i}. Score: {score:.4f} - {chunk['text'][:150]}...")
        
        # Generate (or extract) an answer from the retrieved chunks.
        chunks_list = [chunk for chunk, _ in retrieved]
        if self.use_generative:
            qa_result = self.qa.answer(question, chunks_list)
        else:
            qa_result = self.qa.extract_from_chunks(question, chunks_list)
        
        logger.info(f"QA result - Score: {qa_result['score']:.3f}, Answer length: {len(qa_result.get('answer', ''))}")
        
        # Prepare response
        response = {
            "answer": qa_result["answer"],
            "score": qa_result["score"],
            "source_chunks": qa_result.get("source_chunks", [])
        }
        
        if return_context:
            response["retrieved_chunks"] = [
                {
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": score
                }
                for chunk, score in retrieved
            ]
        
        # Log answer if enabled
        if hasattr(self, 'log_answers') and self.log_answers:
            logger.info(f"Answer: {response['answer']}")
            logger.info(f"Confidence score: {response['score']:.4f}")
        
        return response
    
    def add_documents(self, documents: List[tuple]) -> Dict:
        """
        Incrementally add new documents to the existing index.
        
        Args:
            documents: List of (text, metadata) tuples
            
        Returns:
            Dictionary with statistics
        """
        if not self.is_indexed:
            logger.warning("No existing index. Use index_documents() instead.")
            return self.index_documents()
        
        logger.info(f"Adding {len(documents)} new documents to index...")
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        if not chunks:
            return {"added_chunks": 0, "added_documents": len(documents)}
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vectorstore
        self.vectorstore.add_embeddings(embeddings, chunks)
        
        # Save updated vectorstore
        self.vectorstore.save(self.vectorstore_directory)
        
        stats = {
            "added_chunks": len(chunks),
            "added_documents": len(documents),
            "total_chunks": len(self.vectorstore.chunks)
        }
        
        logger.info(f"Added {len(chunks)} chunks. Total chunks: {stats['total_chunks']}")
        return stats
    
    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        if not self.is_indexed:
            return {"indexed": False}
        
        stats = self.vectorstore.get_stats()
        stats["indexed"] = True
        return stats

