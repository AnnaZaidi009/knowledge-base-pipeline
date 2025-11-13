"""
Ingestion module for document loading, chunking, and indexing.

This module implements the core ingestion pipeline with incremental indexing
capabilities, using LangChain for text splitting and Qdrant for vector storage.
"""

import hashlib
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import DatabaseManager, DocumentMetadata

# Load environment variables
load_dotenv()


class TextProcessor:
    """
    Handles text chunking and preprocessing for document ingestion.
    """
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Initialize the TextProcessor.
        
        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _chunk_text(self, raw_text: str, source_filename: str) -> List[Dict[str, Any]]:
        """
        Split raw text into chunks with metadata.
        
        Args:
            raw_text: The raw text content to chunk.
            source_filename: The source file path/name.
            
        Returns:
            List of dictionaries containing chunk text and metadata.
        """
        chunks = self.splitter.split_text(raw_text)
        
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            # Generate a unique hash for this chunk
            chunk_hash = hashlib.sha256(
                f"{source_filename}_{idx}_{chunk}".encode()
            ).hexdigest()
            
            chunk_data.append({
                "text": chunk,
                "metadata": {
                    "source": source_filename,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "total_chunks": len(chunks)
                }
            })
            
        return chunk_data


class IngestionService:
    """
    Manages the complete document ingestion pipeline with incremental indexing.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        db_manager: DatabaseManager,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the IngestionService.
        
        Args:
            qdrant_client: Qdrant client instance.
            db_manager: Database manager instance.
            collection_name: Name of the Qdrant collection.
            embedding_model_name: Name of the sentence-transformers model.
        """
        self.qdrant_client = qdrant_client
        self.db_manager = db_manager
        self.collection_name = collection_name
        self.text_processor = TextProcessor()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Ensure collection exists
        self._initialize_collection()
        
    def _initialize_collection(self):
        """
        Create Qdrant collection if it doesn't exist.
        """
        try:
            self.qdrant_client.get_collection(self.collection_name)
            # Collection exists, no need to create
            return
        except Exception as e:
            # Collection doesn't exist, create it
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
            except Exception as create_error:
                # If it fails because collection already exists, ignore
                if "already exists" in str(create_error).lower():
                    return
                # Otherwise, re-raise the error
                raise
            
    def _calculate_content_hash(self, raw_content: str) -> str:
        """
        Calculate SHA-256 hash of raw content.
        
        Args:
            raw_content: The raw text content.
            
        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(raw_content.encode()).hexdigest()
        
    def _delete_document_from_qdrant(self, file_path: str):
        """
        Delete all points associated with a file_path from Qdrant.
        
        Args:
            file_path: The file path to filter by.
        """
        try:
            # Delete points where metadata.source matches file_path
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=file_path)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"Warning: Failed to delete existing points for {file_path}: {e}")
            
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
        
    def ingest_document(
        self, 
        file_path: str, 
        raw_content: str
    ) -> Dict[str, Any]:
        """
        Ingest a document with incremental indexing logic.
        
        This method:
        1. Calculates content hash
        2. Checks if document has changed (incremental check)
        3. If changed or new: deletes old embeddings, chunks text, generates embeddings, stores in Qdrant
        4. Updates PostgreSQL metadata
        
        Args:
            file_path: The file path/URI of the document.
            raw_content: The raw text content of the document.
            
        Returns:
            Dictionary with ingestion status and metadata.
        """
        # Calculate content hash
        content_hash = self._calculate_content_hash(raw_content)
        
        # Incremental check: Query database for existing document
        with self.db_manager.get_session() as session:
            existing_doc = self.db_manager.get_document_by_path(session, file_path)
            
            # If document exists and hash matches, skip indexing
            if existing_doc and existing_doc.content_hash == content_hash:
                return {
                    "status": "skipped",
                    "message": f"Document '{file_path}' unchanged, skipping indexing",
                    "file_path": file_path,
                    "content_hash": content_hash,
                    "last_indexed_at": existing_doc.last_indexed_at.isoformat()
                }
            
            # Document is new or modified - proceed with indexing
            # Step 1: Delete existing points from Qdrant
            if existing_doc:
                self._delete_document_from_qdrant(file_path)
            
            # Step 2: Chunk the text
            chunks = self.text_processor._chunk_text(raw_content, file_path)
            
            if not chunks:
                return {
                    "status": "error",
                    "message": "No chunks generated from document",
                    "file_path": file_path
                }
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)
            
            # Step 4: Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=hashlib.sha256(
                        f"{file_path}_{idx}_{datetime.utcnow().isoformat()}".encode()
                    ).hexdigest()[:32],  # Use first 32 chars as UUID-like ID
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": file_path,
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "chunk_hash": chunk["metadata"]["chunk_hash"],
                        "total_chunks": chunk["metadata"]["total_chunks"],
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Step 5: Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Step 6: Update PostgreSQL metadata
            doc_metadata = self.db_manager.upsert_document(
                session=session,
                file_path=file_path,
                content_hash=content_hash
            )
            
            return {
                "status": "indexed" if existing_doc else "created",
                "message": f"Successfully indexed {len(chunks)} chunks from '{file_path}'",
                "file_path": file_path,
                "content_hash": content_hash,
                "chunks_count": len(chunks),
                "last_indexed_at": doc_metadata.last_indexed_at.isoformat()
            }
            
    def bulk_ingest_documents(
        self, 
        documents: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents in bulk.
        
        Args:
            documents: List of tuples (file_path, raw_content).
            
        Returns:
            List of ingestion results for each document.
        """
        results = []
        for file_path, raw_content in documents:
            try:
                result = self.ingest_document(file_path, raw_content)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "message": str(e),
                    "file_path": file_path
                })
        return results
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Qdrant collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }


def create_ingestion_service(
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    db_manager: Optional[DatabaseManager] = None
) -> IngestionService:
    """
    Factory function to create an IngestionService instance.
    
    Args:
        qdrant_host: Qdrant host (defaults to env variable).
        qdrant_port: Qdrant port (defaults to env variable).
        collection_name: Collection name (defaults to env variable).
        db_manager: DatabaseManager instance (creates new if None).
        
    Returns:
        Configured IngestionService instance.
    """
    qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(qdrant_port or os.getenv("QDRANT_PORT", 6333))
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")
    
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    if db_manager is None:
        from database import get_db_manager
        db_manager = get_db_manager()
    
    return IngestionService(
        qdrant_client=qdrant_client,
        db_manager=db_manager,
        collection_name=collection_name
    )
