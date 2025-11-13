"""
Integration tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import Mock, patch, MagicMock
from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_services():
    """Mock services for testing."""
    # Import here to avoid circular imports
    import main
    
    with patch.object(main, 'ingestion_service') as mock_ingestion, \
         patch.object(main, 'rag_retriever') as mock_retriever, \
         patch.object(main, 'db_manager') as mock_db:
        
        # Mock ingestion service
        mock_ingestion.get_collection_stats.return_value = {
            "collection_name": "test",
            "vectors_count": 100,
            "points_count": 100,
            "status": "green"
        }
        mock_ingestion.ingest_document.return_value = {
            "status": "created",
            "message": "Successfully indexed",
            "file_path": "test.txt",
            "content_hash": "abc123",
            "chunks_count": 5,
            "last_indexed_at": "2024-01-01T00:00:00"
        }
        
        # Mock retriever
        from retriever import SearchResult
        mock_result = SearchResult(
            text="Test result",
            source="test.txt",
            score=0.9,
            chunk_index=0,
            metadata={}
        )
        mock_retriever.semantic_search.return_value = [mock_result]
        mock_retriever.answer_question.return_value = {
            "answer": "Test answer",
            "num_sources": 1,
            "sources": [{"source": "test.txt", "score": 0.9}],
            "context_used": ["Test context"]
        }
        mock_retriever.completeness_check.return_value = {
            "topic": "test topic",
            "analysis": "Test analysis",
            "coverage": "partial",
            "num_documents_found": 5,
            "context_reviewed": []
        }
        
        # Mock database manager
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_db.get_all_documents.return_value = []
        # Mock delete_document to return False by default (not found)
        mock_db.delete_document = Mock(return_value=False)
        
        yield {
            "ingestion": mock_ingestion,
            "retriever": mock_retriever,
            "db": mock_db
        }


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client, mock_services):
        """Test successful health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "collection_stats" in data
    
    def test_health_check_failure(self, client):
        """Test health check with service failure."""
        import main
        with patch.object(main, 'ingestion_service') as mock_ingestion:
            mock_ingestion.get_collection_stats.side_effect = Exception("Service error")
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"


class TestIngestEndpoints:
    """Tests for ingestion endpoints."""
    
    def test_ingest_text_success(self, client, mock_services):
        """Test successful text ingestion."""
        response = client.post(
            "/ingest/text",
            json={
                "file_path": "test.txt",
                "content": "Test document content"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "file_path" in data
        assert "chunks_count" in data
    
    def test_ingest_text_missing_fields(self, client):
        """Test text ingestion with missing fields."""
        response = client.post(
            "/ingest/text",
            json={"file_path": "test.txt"}  # Missing content
        )
        assert response.status_code == 422  # Validation error
    
    def test_ingest_file_success(self, client, mock_services):
        """Test successful file upload."""
        response = client.post(
            "/ingest/file",
            files={"file": ("test.txt", "Test file content", "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
    
    def test_ingest_file_invalid_encoding(self, client, mock_services):
        """Test file upload with invalid encoding."""
        # Test with binary file that can't be decoded as UTF-8
        # The endpoint should handle this gracefully
        response = client.post(
            "/ingest/file",
            files={"file": ("test.bin", b"\xff\xfe\x00\x01", "application/octet-stream")}
        )
        # Should return 400 for invalid encoding
        assert response.status_code == 400


class TestSearchEndpoints:
    """Tests for search endpoints."""
    
    def test_semantic_search_success(self, client, mock_services):
        """Test successful semantic search."""
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "top_k": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "num_results" in data
        assert len(data["results"]) > 0
    
    def test_semantic_search_missing_query(self, client):
        """Test search with missing query."""
        response = client.post(
            "/search",
            json={"top_k": 5}  # Missing query
        )
        assert response.status_code == 422  # Validation error
    
    def test_qa_success(self, client, mock_services):
        """Test successful Q&A."""
        response = client.post(
            "/query/qa",
            json={
                "question": "What is machine learning?",
                "top_k": 5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "num_sources" in data
    
    def test_completeness_check_success(self, client, mock_services):
        """Test successful completeness check."""
        response = client.post(
            "/query/completeness",
            json={
                "topic": "machine learning",
                "top_k": 10
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "topic" in data
        assert "analysis" in data
        assert "coverage" in data
        assert "num_documents_found" in data


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_stats_endpoint(self, client, mock_services):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "collection_name" in data
        assert "vectors_count" in data
        assert "total_documents" in data


class TestDeleteEndpoint:
    """Tests for delete endpoint."""
    
    def test_delete_document_success(self, client, mock_services):
        """Test successful document deletion."""
        # Mock database delete to return True (success)
        mock_services["db"].delete_document.return_value = True
        
        response = client.delete("/documents/test.txt")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
    
    def test_delete_document_not_found(self, client, mock_services):
        """Test deleting non-existent document."""
        # Mock database delete to return False (not found)
        mock_services["db"].delete_document.return_value = False
        
        response = client.delete("/documents/nonexistent.txt")
        assert response.status_code == 404

