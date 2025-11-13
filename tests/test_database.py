"""
Unit tests for database operations.
"""
import pytest
from datetime import datetime
from database import (
    DatabaseManager,
    DocumentMetadata,
    get_db_manager
)


class TestDatabaseManager:
    """Tests for DatabaseManager class."""
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        
        assert manager.database_url == db_url
        assert manager.engine is not None
        assert manager.SessionLocal is not None
    
    def test_create_tables(self):
        """Test table creation."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        # Tables should be created without error
        # Verify by trying to query
        with manager.get_session() as session:
            count = session.query(DocumentMetadata).count()
            assert count == 0  # Should be empty initially
    
    def test_get_session_context_manager(self):
        """Test session context manager."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            assert session is not None
            # Session should be usable
            count = session.query(DocumentMetadata).count()
            assert isinstance(count, int)
    
    def test_upsert_document_new(self):
        """Test upserting a new document."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            doc = manager.upsert_document(
                session=session,
                file_path="test.txt",
                content_hash="abc123"
            )
            
            assert doc.file_path == "test.txt"
            assert doc.content_hash == "abc123"
            assert doc.id is not None
            assert doc.last_indexed_at is not None
    
    def test_upsert_document_update(self):
        """Test updating an existing document."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            # Create initial document
            doc1 = manager.upsert_document(
                session=session,
                file_path="test.txt",
                content_hash="abc123"
            )
            original_time = doc1.last_indexed_at
            
            # Update document
            import time
            time.sleep(0.1)  # Ensure timestamp difference
            
            doc2 = manager.upsert_document(
                session=session,
                file_path="test.txt",
                content_hash="def456"
            )
            
            assert doc2.id == doc1.id  # Same document
            assert doc2.content_hash == "def456"  # Updated hash
            assert doc2.last_indexed_at > original_time  # Updated timestamp
    
    def test_get_document_by_path_exists(self):
        """Test getting document that exists."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            # Create document
            manager.upsert_document(
                session=session,
                file_path="test.txt",
                content_hash="abc123"
            )
            
            # Retrieve document
            doc = manager.get_document_by_path(session, "test.txt")
            
            assert doc is not None
            assert doc.file_path == "test.txt"
            assert doc.content_hash == "abc123"
    
    def test_get_document_by_path_not_exists(self):
        """Test getting document that doesn't exist."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            doc = manager.get_document_by_path(session, "nonexistent.txt")
            assert doc is None
    
    def test_get_all_documents(self):
        """Test getting all documents."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            # Create multiple documents
            manager.upsert_document(session, "doc1.txt", "hash1")
            manager.upsert_document(session, "doc2.txt", "hash2")
            manager.upsert_document(session, "doc3.txt", "hash3")
            
            # Get all documents
            docs = manager.get_all_documents(session)
            
            assert len(docs) == 3
            file_paths = [doc.file_path for doc in docs]
            assert "doc1.txt" in file_paths
            assert "doc2.txt" in file_paths
            assert "doc3.txt" in file_paths
    
    def test_delete_document_exists(self):
        """Test deleting an existing document."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            # Create document
            manager.upsert_document(session, "test.txt", "abc123")
            
            # Delete document
            deleted = manager.delete_document(session, "test.txt")
            
            assert deleted == True
            
            # Verify deleted
            doc = manager.get_document_by_path(session, "test.txt")
            assert doc is None
    
    def test_delete_document_not_exists(self):
        """Test deleting a non-existent document."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(database_url=db_url)
        manager.create_tables()
        
        with manager.get_session() as session:
            deleted = manager.delete_document(session, "nonexistent.txt")
            assert deleted == False


class TestGetDBManager:
    """Tests for get_db_manager function."""
    
    def test_get_db_manager_singleton(self):
        """Test that get_db_manager returns singleton."""
        import database
        database.db_manager = None  # Reset
        
        manager1 = get_db_manager()
        manager2 = get_db_manager()
        
        assert manager1 is manager2  # Should be same instance

