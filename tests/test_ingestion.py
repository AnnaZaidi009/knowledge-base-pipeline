"""
Unit tests for ingestion service.
"""
import pytest
import hashlib
from unittest.mock import Mock, patch, MagicMock
from ingestion import IngestionService, TextProcessor, create_ingestion_service


class TestTextProcessor:
    """Tests for TextProcessor class."""
    
    def test_text_processor_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor(chunk_size=512, chunk_overlap=100)
        assert processor.chunk_size == 512
        assert processor.chunk_overlap == 100
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 10  # ~250 chars
        chunks = processor._chunk_text(text, "test.txt")
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all(chunk["metadata"]["source"] == "test.txt" for chunk in chunks)
    
    def test_chunk_text_metadata(self):
        """Test chunk metadata structure."""
        processor = TextProcessor()
        text = "Short text"
        chunks = processor._chunk_text(text, "test.txt")
        
        assert len(chunks) > 0
        chunk = chunks[0]
        assert "chunk_index" in chunk["metadata"]
        assert "chunk_hash" in chunk["metadata"]
        assert "total_chunks" in chunk["metadata"]
        assert chunk["metadata"]["total_chunks"] == len(chunks)


class TestIngestionService:
    """Tests for IngestionService class."""
    
    def test_ingestion_service_initialization(self, mock_qdrant_client, mock_db_manager):
        """Test IngestionService initialization."""
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test_collection"
        )
        
        assert service.qdrant_client == mock_qdrant_client
        assert service.db_manager == mock_db_manager
        assert service.collection_name == "test_collection"
    
    def test_calculate_content_hash(self, mock_qdrant_client, mock_db_manager):
        """Test content hash calculation."""
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test"
        )
        
        content = "test content"
        hash1 = service._calculate_content_hash(content)
        hash2 = service._calculate_content_hash(content)
        hash3 = service._calculate_content_hash("different content")
        
        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 produces 64 char hex string
    
    @patch('ingestion.SentenceTransformer')
    def test_ingest_new_document(self, mock_transformer, mock_qdrant_client, mock_db_manager, sample_text):
        """Test ingesting a new document."""
        import numpy as np
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384] * 3)  # 3 chunks, 384 dims
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test"
        )
        service.embedding_model = mock_model
        
        result = service.ingest_document("test.txt", sample_text)
        
        assert result["status"] in ["created", "indexed"]
        assert "file_path" in result
        assert "content_hash" in result
        assert "chunks_count" in result
        assert result["chunks_count"] > 0
        mock_qdrant_client.upsert.assert_called_once()
    
    @patch('ingestion.SentenceTransformer')
    def test_ingest_unchanged_document(self, mock_transformer, mock_qdrant_client, mock_db_manager, sample_text):
        """Test ingesting an unchanged document (should skip)."""
        import numpy as np
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1] * 384] * 3)  # 3 chunks, 384 dims
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test"
        )
        service.embedding_model = mock_model
        
        # First ingestion
        result1 = service.ingest_document("test.txt", sample_text)
        assert result1["status"] in ["created", "indexed"]
        
        # Second ingestion (unchanged) - should skip because hash matches
        result2 = service.ingest_document("test.txt", sample_text)
        assert result2["status"] == "skipped"
        assert "unchanged" in result2["message"].lower()
    
    @patch('ingestion.SentenceTransformer')
    def test_ingest_empty_document(self, mock_transformer, mock_qdrant_client, mock_db_manager):
        """Test ingesting an empty document."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test"
        )
        service.embedding_model = mock_model
        
        result = service.ingest_document("empty.txt", "")
        
        assert result["status"] == "error"
        assert "no chunks" in result["message"].lower()
    
    @patch('ingestion.SentenceTransformer')
    def test_get_collection_stats(self, mock_transformer, mock_qdrant_client, mock_db_manager):
        """Test getting collection statistics."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        # Mock collection info
        mock_collection = Mock()
        mock_collection.vectors_count = 100
        mock_collection.points_count = 100
        mock_collection.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_collection
        
        service = IngestionService(
            qdrant_client=mock_qdrant_client,
            db_manager=mock_db_manager,
            collection_name="test"
        )
        service.embedding_model = mock_model
        
        stats = service.get_collection_stats()
        
        assert stats["collection_name"] == "test"
        assert stats["vectors_count"] == 100
        assert stats["points_count"] == 100


class TestCreateIngestionService:
    """Tests for create_ingestion_service factory function."""
    
    @patch('ingestion.QdrantClient')
    @patch('ingestion.SentenceTransformer')
    def test_create_ingestion_service(self, mock_transformer, mock_qdrant, mock_db_manager):
        """Test creating ingestion service via factory."""
        mock_qdrant.return_value = Mock()
        mock_transformer.return_value = Mock(get_sentence_embedding_dimension=Mock(return_value=384))
        
        service = create_ingestion_service(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="test",
            db_manager=mock_db_manager
        )
        
        assert isinstance(service, IngestionService)
        mock_qdrant.assert_called_once_with(host="localhost", port=6333)

