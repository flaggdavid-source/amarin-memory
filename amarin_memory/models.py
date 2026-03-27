"""SQLAlchemy models for the Amarin memory system."""

import datetime
from sqlalchemy import String, Text, Float, ForeignKey, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column

from amarin_memory.database import Base


class MemoryBlock(Base):
    """Core memory block — always visible in context."""
    __tablename__ = "memory_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String(100), unique=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class ArchivalMemory(Base):
    """Long-term searchable memory storage with semantic search."""
    __tablename__ = "archival_memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[str] = mapped_column(String(500), default="")
    embedding: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    member_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, default=None, index=True
    )
    importance: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=0.5
    )
    last_accessed: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True, default=None
    )
    emotion_tags: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default=None
    )
    surprise: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    is_active: Mapped[int] = mapped_column(default=1)
    protected: Mapped[int] = mapped_column(default=0)

    __table_args__ = (
        Index("ix_archival_member_importance", "member_id", "importance"),
    )


class CoreBlockSnapshot(Base):
    """Snapshot of a core memory block before modification."""
    __tablename__ = "core_block_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    block_label: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[str] = mapped_column(Text)
    trigger: Mapped[str] = mapped_column(String(30))
    snapshot_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class MemoryEdit(Base):
    """Audit trail for memory curation operations."""
    __tablename__ = "memory_edits"

    id: Mapped[int] = mapped_column(primary_key=True)
    memory_id: Mapped[int] = mapped_column(ForeignKey("archival_memories.id"))
    action: Mapped[str] = mapped_column(String(20))
    original_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )


class SessionSummary(Base):
    """Session summary — narrative diary entries about conversations."""
    __tablename__ = "session_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    summary: Mapped[str] = mapped_column(Text)
    message_count: Mapped[int] = mapped_column(default=0)
    key_topics: Mapped[str] = mapped_column(String(500), default="")
    emotional_arc: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    period_type: Mapped[str] = mapped_column(String(20), default="session")
    period_start: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True, default=None
    )
    period_end: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True, default=None
    )
