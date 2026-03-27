"""Basic smoke tests for amarin-memory package."""

import datetime
import pytest
from sqlalchemy import text

from amarin_memory import (
    create_memory_engine,
    create_session,
    Base,
    MemoryBlock,
    ArchivalMemory,
    MemoryEngine,
)
from amarin_memory.memory import (
    get_core_blocks,
    get_block,
    upsert_block,
    add_archival,
    search_archival,
    list_archival,
    get_archival_by_id,
    revise_archival,
    deactivate_archival,
    activate_archival,
    protect_archival,
    release_archival,
    build_memory_context,
    build_archival_context,
    _merge_tags,
)
from amarin_memory.decay import apply_temporal_decay


@pytest.fixture
def db():
    """Create an in-memory SQLite database with all tables."""
    engine = create_memory_engine(":memory:")
    Base.metadata.create_all(engine)
    # Create vec_archival virtual table
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_archival "
            "USING vec0(embedding float[768])"
        ))
        conn.commit()
    session_factory = create_session(engine)
    session = session_factory()
    yield session
    session.close()


class TestCoreBlocks:
    def test_upsert_and_get(self, db):
        block = upsert_block(db, "persona", "I am a test character.")
        assert block.label == "persona"
        assert block.value == "I am a test character."

        retrieved = get_block(db, "persona")
        assert retrieved is not None
        assert retrieved.value == "I am a test character."

    def test_get_all_blocks(self, db):
        upsert_block(db, "persona", "Test persona")
        upsert_block(db, "human", "Test human")
        blocks = get_core_blocks(db)
        assert len(blocks) == 2
        labels = [b["label"] for b in blocks]
        assert "persona" in labels
        assert "human" in labels

    def test_upsert_updates_existing(self, db):
        upsert_block(db, "persona", "Version 1")
        upsert_block(db, "persona", "Version 2")
        block = get_block(db, "persona")
        assert block.value == "Version 2"

    def test_build_memory_context(self, db):
        upsert_block(db, "persona", "I am helpful.")
        ctx = build_memory_context(db)
        assert "Core Memories" in ctx
        assert "I am helpful." in ctx


class TestArchivalMemory:
    def test_add_and_search(self, db):
        add_archival(db, "The dragon sleeps in the eastern cave.", tags="dragon,cave")
        results = search_archival(db, "dragon")
        assert len(results) == 1
        assert "dragon" in results[0]["content"].lower()

    def test_list(self, db):
        add_archival(db, "Memory one", tags="test")
        add_archival(db, "Memory two", tags="test")
        results = list_archival(db)
        assert len(results) == 2

    def test_get_by_id(self, db):
        mem = add_archival(db, "Specific memory", tags="specific")
        retrieved = get_archival_by_id(db, mem.id)
        assert retrieved is not None
        assert retrieved.content == "Specific memory"

    def test_revise(self, db):
        mem = add_archival(db, "Original content", tags="test")
        result = revise_archival(db, mem.id, "Revised content", reason="test revision")
        assert result["revised"] is True
        updated = get_archival_by_id(db, mem.id)
        assert updated.content == "Revised content"

    def test_soft_delete_and_restore(self, db):
        mem = add_archival(db, "To be forgotten", tags="test")
        result = deactivate_archival(db, mem.id, reason="test forget")
        assert result["forgotten"] is True

        # Should not appear in search
        results = search_archival(db, "forgotten")
        assert len(results) == 0

        # Restore
        result = activate_archival(db, mem.id)
        assert result["restored"] is True
        results = search_archival(db, "forgotten")
        assert len(results) == 1

    def test_protect_and_release(self, db):
        mem = add_archival(db, "Protected memory", tags="test", importance=0.3)
        result = protect_archival(db, mem.id)
        assert result["protected"] is True
        updated = get_archival_by_id(db, mem.id)
        assert updated.protected == 1
        assert updated.importance >= 0.7

        result = release_archival(db, mem.id)
        assert result["released"] is True
        updated = get_archival_by_id(db, mem.id)
        assert updated.protected == 0


class TestDecay:
    def test_decay_skips_recent(self, db):
        mem = add_archival(db, "Recent memory", importance=0.8)
        mem.last_accessed = datetime.datetime.now(datetime.timezone.utc)
        db.commit()
        apply_temporal_decay(db)
        db.refresh(mem)
        assert mem.importance == 0.8  # Should not decay

    def test_decay_reduces_old(self, db):
        mem = add_archival(db, "Old memory", importance=0.8)
        mem.last_accessed = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
        db.commit()
        apply_temporal_decay(db, decay_rate=0.05)
        db.refresh(mem)
        assert mem.importance < 0.8

    def test_decay_skips_protected(self, db):
        mem = add_archival(db, "Protected old memory", importance=0.8)
        mem.last_accessed = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
        mem.protected = 1
        db.commit()
        apply_temporal_decay(db, decay_rate=0.05)
        db.refresh(mem)
        assert mem.importance == 0.8


class TestHelpers:
    def test_merge_tags(self):
        assert _merge_tags("a,b", "b,c") == "a,b,c"
        assert _merge_tags("", "x,y") == "x,y"
        assert _merge_tags("x", "") == "x"

    def test_build_archival_context_empty(self):
        assert build_archival_context([]) == ""

    def test_build_archival_context_formats(self):
        memories = [
            {"content": "Test memory content", "score": 0.8},
        ]
        ctx = build_archival_context(memories)
        assert "Retrieved Memories" in ctx
        assert "Test memory content" in ctx


class TestMemoryEngine:
    def test_init_db(self):
        engine = MemoryEngine(db_path=":memory:")
        engine.init_db()
        # Should be able to get an empty block list
        assert engine.get_blocks() == []

    def test_set_and_get_block(self):
        engine = MemoryEngine(db_path=":memory:")
        engine.init_db()
        engine.set_block("test", "hello world")
        blocks = engine.get_blocks()
        assert len(blocks) == 1
        assert blocks[0]["label"] == "test"
        assert blocks[0]["value"] == "hello world"
