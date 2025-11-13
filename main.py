"""
FastAPI application for AI-Powered Knowledge Base Search & Enrichment.

This module provides RESTful API endpoints for document ingestion,
semantic search, and RAG-based question answering.
"""

import os
import requests
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import aiofiles

from ingestion import create_ingestion_service, IngestionService
from retriever import create_rag_retriever, RAGRetriever
from database import get_db_manager, DatabaseManager

# Load environment variables
load_dotenv()

# Global service instances
ingestion_service: Optional[IngestionService] = None
rag_retriever: Optional[RAGRetriever] = None
db_manager: Optional[DatabaseManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.
    """
    # Startup: Initialize services
    global ingestion_service, rag_retriever, db_manager
    
    try:
        db_manager = get_db_manager()
        ingestion_service = create_ingestion_service(db_manager=db_manager)
        rag_retriever = create_rag_retriever()
        
        print("✓ Database connection established")
        print("✓ Ingestion service initialized")
        print("✓ RAG retriever initialized")
        print(f"✓ Collection: {os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base')}")
        
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup resources
    print("Shutting down services...")


# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Knowledge Base API",
    description="RAG pipeline for document ingestion, semantic search, and Q&A",
    version="1.0.0",
    lifespan=lifespan
)


# =====================
# Pydantic Models
# =====================

class IngestTextRequest(BaseModel):
    """Request model for text ingestion."""
    file_path: str = Field(..., description="Unique identifier/path for the document")
    content: str = Field(..., description="Raw text content to ingest")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "docs/introduction.txt",
                "content": "This is a sample document about machine learning..."
            }
        }


class IngestResponse(BaseModel):
    """Response model for ingestion operations."""
    status: str
    message: str
    file_path: str
    content_hash: Optional[str] = None
    chunks_count: Optional[int] = None
    last_indexed_at: Optional[str] = None


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "top_k": 5
            }
        }


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    num_sources: int
    sources: List[dict]
    context_used: Optional[List[str]] = None


class CompletenessRequest(BaseModel):
    """Request model for completeness check."""
    topic: str = Field(..., description="The topic to analyze for completeness")
    top_k: int = Field(10, ge=1, le=50, description="Number of documents to review")
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "neural networks",
                "top_k": 10
            }
        }


class CompletenessResponse(BaseModel):
    """Response model for completeness check."""
    topic: str
    analysis: str
    coverage: str
    num_documents_found: int
    context_reviewed: List[dict]


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "deep learning frameworks",
                "top_k": 5
            }
        }


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str
    results: List[dict]
    num_results: int


# =====================
# API Endpoints
# =====================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI-Powered Knowledge Base API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "ingest_text": "POST /ingest/text",
            "ingest_file": "POST /ingest/file",
            "search": "POST /search",
            "qa": "POST /query/qa",
            "completeness": "POST /query/completeness",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = ingestion_service.get_collection_stats()
        return {
            "status": "healthy",
            "services": {
                "database": "connected",
                "qdrant": "connected",
                "ingestion": "ready",
                "retrieval": "ready"
            },
            "collection_stats": stats
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    """
    Ingest text content with incremental indexing.
    
    This endpoint:
    - Accepts text content with a file path identifier
    - Performs hash-based change detection
    - Only indexes if content has changed (incremental indexing)
    - Returns ingestion status
    """
    try:
        result = ingestion_service.ingest_document(
            file_path=request.file_path,
            raw_content=request.content
        )
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file upload.
    
    This endpoint:
    - Accepts file uploads
    - Reads file content
    - Processes through the ingestion pipeline
    """
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Ingest the document
        result = ingestion_service.ingest_document(
            file_path=file.filename,
            raw_content=text_content
        )
        
        return IngestResponse(**result)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be a valid UTF-8 text file"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on the knowledge base.
    
    This endpoint:
    - Accepts a search query
    - Returns semantically similar documents
    - Ranks results by relevance score
    """
    try:
        search_results = rag_retriever.semantic_search(
            query=request.query,
            top_k=request.top_k
        )
        
        results = [
            {
                "text": result.text,
                "source": result.source,
                "score": result.score,
                "chunk_index": result.chunk_index
            }
            for result in search_results
        ]
        
        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/qa", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint:
    - Performs semantic search to retrieve relevant context
    - Uses LLM to generate an answer based on the context
    - Returns the answer with supporting evidence
    """
    try:
        result = rag_retriever.answer_question(
            question=request.question,
            top_k=request.top_k
        )
        
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/completeness", response_model=CompletenessResponse)
async def completeness_check(request: CompletenessRequest):
    """
    Analyze knowledge base completeness for a topic.
    
    This endpoint:
    - Retrieves documents related to the topic
    - Analyzes coverage and identifies gaps
    - Provides recommendations for improvement
    """
    try:
        result = rag_retriever.completeness_check(
            topic=request.topic,
            top_k=request.top_k
        )
        
        return CompletenessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    try:
        vectors_count = 0
        doc_count = 0
        recent_docs_list = []
        
        try:
            response = requests.get("http://localhost:6333/collections/knowledge_base")
            if response.status_code == 200:
                data = response.json()
                vectors_count = data.get("result", {}).get("points_count", 0)
        except:
            pass
        
        try:
            with db_manager.get_session() as session:
                all_docs = db_manager.get_all_documents(session)
                doc_count = len(all_docs)
                recent_docs = sorted(all_docs, key=lambda x: x.last_indexed_at, reverse=True)[:5]
                recent_docs_list = [
                    {"file_path": doc.file_path, "last_indexed_at": doc.last_indexed_at.isoformat()}
                    for doc in recent_docs
                ]
        except:
            pass
            
        return {
            "collection_name": os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base'),
            "vectors_count": vectors_count,
            "total_documents": doc_count,
            "recent_documents": recent_docs_list
        }
    except:
        return {
            "collection_name": os.getenv('QDRANT_COLLECTION_NAME', 'knowledge_base'),
            "vectors_count": 0,
            "total_documents": 0,
            "recent_documents": []
        }


@app.delete("/documents/{file_path:path}")
async def delete_document(file_path: str):
    """
    Delete a document from the knowledge base.
    
    This endpoint:
    - Removes document from Qdrant vector store
    - Removes metadata from PostgreSQL
    """
    try:
        # Delete from database
        with db_manager.get_session() as session:
            deleted = db_manager.delete_document(session, file_path)
            
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from Qdrant
        ingestion_service._delete_document_from_qdrant(file_path)
        
        return {
            "status": "deleted",
            "message": f"Document '{file_path}' has been removed",
            "file_path": file_path
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Error Handlers
# =====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# =====================
# Main Entry Point
# =====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  AI-Powered Knowledge Base API                            ║
    ║  Version: 1.0.0                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    🚀 Starting server at http://{host}:{port}
    📚 API Documentation: http://{host}:{port}/docs
    📊 Interactive API: http://{host}:{port}/redoc
    """)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
