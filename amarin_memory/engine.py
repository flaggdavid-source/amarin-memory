"""MemoryEngine — high-level convenience wrapper for the amarin-memory package.

Usage:
    from amarin_memory import MemoryEngine

    engine = MemoryEngine(db_path="world.db", embedding_url="http://localhost:8200")
    engine.init_db()

    # Store with deduplication
    result = await engine.store("The barkeep mentioned a dungeon to the east.", tags="rumor,dungeon")

    # Semantic search
    results = await engine.search("dungeon", limit=5)

    # Apply temporal decay
    engine.apply_decay()
"""

from sqlalchemy.orm import Session

from amarin_memory.database import create_memory_engine, create_session, Base
from amarin_memory.decay import apply_temporal_decay
from amarin_memory.memory import (
    # Vector ops
    sync_vec_table,
    # Search
    semantic_search,
    find_similar_memories,
    retrieve_memories,
    # Dedup
    deduplicate_and_save,
    # Core blocks
    get_core_blocks,
    get_block,
    upsert_block,
    get_block_history,
    restore_block,
    get_member_core_memory,
    delete_block,
    # Archival CRUD
    add_archival,
    add_archival_with_embedding,
    search_archival,
    list_archival,
    search_archival_multi,
    get_archival_by_id,
    # Curation
    revise_archival,
    deactivate_archival,
    activate_archival,
    protect_archival,
    release_archival,
    log_memory_edit,
    search_archival_for_review,
    get_memory_edits,
    # Summaries
    get_recent_summaries,
    get_recent_agent_summaries,
    # Context
    build_archival_context,
    build_memory_context,
)


class MemoryEngine:
    """High-level wrapper that owns a database connection and embedding config.

    Designed for consumers who just want to ``engine.store()`` and ``engine.search()``
    without managing sessions or passing URLs everywhere.
    """

    def __init__(self, db_path: str, embedding_url: str = "http://localhost:8200"):
        self.db_path = db_path
        self.embedding_url = embedding_url
        self._engine = create_memory_engine(db_path)
        self._session_factory = create_session(self._engine)

    def init_db(self):
        """Create all tables (safe to call repeatedly)."""
        from sqlalchemy import text as sa_text
        Base.metadata.create_all(self._engine)
        # Create vec_archival virtual table if it doesn't exist
        with self._engine.connect() as conn:
            conn.execute(sa_text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_archival "
                "USING vec0(embedding float[768])"
            ))
            conn.commit()

    def get_session(self) -> Session:
        """Get a new database session. Caller must close it."""
        return self._session_factory()

    # -- High-level async API ------------------------------------------------

    async def store(
        self,
        content: str,
        tags: str = "",
        member_id: str | None = None,
        importance: float = 0.5,
        emotion_tags: str | None = None,
    ) -> dict:
        """Store a memory with deduplication and surprise scoring."""
        db = self.get_session()
        try:
            return await deduplicate_and_save(
                db, content, tags, member_id, importance, emotion_tags,
                embedding_url=self.embedding_url,
            )
        finally:
            db.close()

    async def search(
        self,
        query: str,
        limit: int = 10,
        member_id: str | None = None,
    ) -> list[dict] | None:
        """Semantic search. Returns None if embedding service is down."""
        db = self.get_session()
        try:
            return await semantic_search(
                db, query, limit, member_id,
                embedding_url=self.embedding_url,
            )
        finally:
            db.close()

    async def retrieve(
        self,
        message: str,
        keywords: list[str],
        limit: int = 5,
        member_id: str | None = None,
    ) -> tuple[list[dict], str]:
        """Retrieve memories with semantic→keyword fallback."""
        db = self.get_session()
        try:
            return await retrieve_memories(
                db, message, keywords, limit, member_id,
                embedding_url=self.embedding_url,
            )
        finally:
            db.close()

    # -- Synchronous helpers -------------------------------------------------

    def apply_decay(self, decay_rate: float = 0.01, min_importance: float = 0.1):
        """Apply temporal decay to all memories."""
        db = self.get_session()
        try:
            apply_temporal_decay(db, decay_rate, min_importance)
        finally:
            db.close()

    def sync_vectors(self):
        """Sync any un-indexed embeddings into the vec table."""
        db = self.get_session()
        try:
            sync_vec_table(db)
        finally:
            db.close()

    def get_blocks(self, member_id: str | None = None) -> list[dict]:
        """Get core memory blocks, optionally filtered by member_id."""
        db = self.get_session()
        try:
            return get_core_blocks(db, member_id=member_id)
        finally:
            db.close()

    def set_block(self, label: str, value: str, trigger: str = "api_update", member_id: str | None = None):
        """Upsert a core memory block."""
        db = self.get_session()
        try:
            return upsert_block(db, label, value, trigger, member_id=member_id)
        finally:
            db.close()

    def build_context(self, member_id: str | None = None) -> str:
        """Build core memory context string for prompt injection."""
        db = self.get_session()
        try:
            return build_memory_context(db, member_id=member_id)
        finally:
            db.close()
