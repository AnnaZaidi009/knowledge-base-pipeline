"""
Database module for managing document metadata and PostgreSQL connections.

This module provides utilities for persisting document metadata to enable
incremental indexing and change detection in the RAG pipeline.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# SQLAlchemy Base
Base = declarative_base()


class DocumentMetadata(Base):
    """
    SQLAlchemy model for storing document metadata.

    Attributes:
        id: Primary key (UUID).
        file_path: Original path/URI of the document.
        content_hash: SHA-256 hash of the raw file content for change detection.
        last_indexed_at: Timestamp of last successful indexing.
    """

    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(String, unique=True, nullable=False, index=True)
    content_hash = Column(String, nullable=False)
    last_indexed_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DocumentMetadataSchema(BaseModel):
    """
    Pydantic schema for DocumentMetadata validation and serialization.
    """

    id: uuid.UUID
    file_path: str
    content_hash: str
    last_indexed_at: datetime

    class Config:
        from_attributes = True


class DatabaseManager:
    """
    Utility class for managing PostgreSQL database connections and operations.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the DatabaseManager.

        Args:
            database_url: PostgreSQL connection string. If None, uses POSTGRES_DSN from .env
        """
        self.database_url = database_url or os.getenv("POSTGRES_DSN")
        if not self.database_url:
            raise ValueError(
                "Database URL must be provided or set in POSTGRES_DSN environment variable"
            )

        # SQLite doesn't support pool_size and max_overflow
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url, pool_pre_ping=True
            )
        else:
            self.engine = create_engine(
                self.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20
            )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """
        Create all database tables defined in Base.
        """
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.

        Yields:
            Session: SQLAlchemy database session.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_document_by_path(
        self, session: Session, file_path: str
    ) -> Optional[DocumentMetadata]:
        """
        Retrieve a document metadata record by file path.

        Args:
            session: SQLAlchemy session.
            file_path: The file path to search for.

        Returns:
            DocumentMetadata object if found, None otherwise.
        """
        return (
            session.query(DocumentMetadata)
            .filter(DocumentMetadata.file_path == file_path)
            .first()
        )

    def upsert_document(
        self, session: Session, file_path: str, content_hash: str
    ) -> DocumentMetadata:
        """
        Create or update a document metadata record.

        Args:
            session: SQLAlchemy session.
            file_path: The file path.
            content_hash: SHA-256 hash of the document content.

        Returns:
            The created or updated DocumentMetadata object.
        """
        existing = self.get_document_by_path(session, file_path)

        if existing:
            existing.content_hash = content_hash
            existing.last_indexed_at = datetime.utcnow()
            session.flush()
            return existing
        else:
            new_doc = DocumentMetadata(
                file_path=file_path,
                content_hash=content_hash,
                last_indexed_at=datetime.utcnow(),
            )
            session.add(new_doc)
            session.flush()
            return new_doc

    def get_all_documents(self, session: Session) -> List[DocumentMetadata]:
        """
        Retrieve all document metadata records.

        Args:
            session: SQLAlchemy session.

        Returns:
            List of all DocumentMetadata objects.
        """
        return session.query(DocumentMetadata).all()

    def delete_document(self, session: Session, file_path: str) -> bool:
        """
        Delete a document metadata record by file path.

        Args:
            session: SQLAlchemy session.
            file_path: The file path to delete.

        Returns:
            True if deleted, False if not found.
        """
        existing = self.get_document_by_path(session, file_path)
        if existing:
            session.delete(existing)
            session.flush()
            return True
        return False


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create the global DatabaseManager instance.

    Returns:
        DatabaseManager instance.
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        db_manager.create_tables()
    return db_manager
