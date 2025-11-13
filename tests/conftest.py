"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment variables
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("QDRANT_COLLECTION_NAME", "test_knowledge_base")
os.environ.setdefault("POSTGRES_DSN", "sqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = Mock(spec=QdrantClient)
    client.get_collection = Mock(return_value=Mock(vectors_count=0, points_count=0, status="green"))
    client.create_collection = Mock()
    client.search = Mock(return_value=[])
    client.upsert = Mock()
    client.delete = Mock()
    return client


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    from database import DatabaseManager
    
    # Use in-memory SQLite for testing
    db_url = "sqlite:///:memory:"
    manager = DatabaseManager(database_url=db_url)
    manager.create_tables()
    return manager


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from data. There are three main types: supervised learning, unsupervised
    learning, and reinforcement learning. Supervised learning uses labeled data to train
    models, while unsupervised learning finds patterns in unlabeled data. Reinforcement
    learning involves agents learning through trial and error in an environment.
    """


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "file_path": "test_doc.txt",
        "content": """
        Deep learning is a subset of machine learning that uses neural networks with
        multiple layers. Convolutional neural networks (CNNs) are used for image processing,
        while recurrent neural networks (RNNs) are used for sequential data. Transformers
        have revolutionized natural language processing with attention mechanisms.
        """
    }

