"""Core memory functions — semantic search, deduplication, CRUD, context building."""

import json
import struct
import datetime
import logging
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from amarin_memory.models import (
    MemoryBlock,
    ArchivalMemory,
    MemoryEdit,
    CoreBlockSnapshot,
    SessionSummary,
)
from amarin_memory.embeddings import get_embeddings, get_query_embedding

logger = logging.getLogger("amarin_memory")


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two normalized vectors (just a dot product)."""
    return float(np.dot(a, b))


def _pack_vec(vec: list[float]) -> bytes:
    """Pack a float list into little-endian binary for sqlite-vec."""
    return struct.pack(f"<{len(vec)}f", *vec)


def sync_vec_table(db: Session):
    """Sync the vec_archival virtual table from archival_memories embeddings.

    Inserts any rows that have embeddings but aren't yet in vec_archival.
    Safe to call repeatedly (idempotent).
    """
    conn = db.connection()
    rows = conn.execute(text(
        "SELECT a.id, a.embedding FROM archival_memories a "
        "WHERE a.embedding IS NOT NULL "
        "AND a.id NOT IN (SELECT rowid FROM vec_archival)"
    )).fetchall()

    count = 0
    for row_id, embedding_json in rows:
        try:
            vec = json.loads(embedding_json)
            conn.execute(
                text("INSERT INTO vec_archival(rowid, embedding) VALUES (:id, :emb)"),
                {"id": row_id, "emb": _pack_vec(vec)},
            )
            count += 1
        except (json.JSONDecodeError, TypeError, struct.error):
            logger.warning("Skipping bad embedding for archival memory %d", row_id)
    if count:
        db.commit()
        logger.info("Synced %d embeddings to vec_archival", count)


def insert_vec_embedding(db: Session, memory_id: int, embedding: list[float]):
    """Insert a single embedding into the vec_archival table."""
    conn = db.connection()
    conn.execute(
        text("INSERT OR REPLACE INTO vec_archival(rowid, embedding) VALUES (:id, :emb)"),
        {"id": memory_id, "emb": _pack_vec(embedding)},
    )
    db.commit()


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

async def semantic_search(
    db: Session,
    query: str,
    limit: int = 10,
    member_id: str | None = None,
    embedding_url: str | None = None,
) -> Optional[list[dict]]:
    """Search archival memories by semantic similarity using sqlite-vec.

    Returns None if the embedding service is unavailable (caller should fall back).
    Returns an empty list if the service is up but no memories match.
    """
    kwargs = {"base_url": embedding_url} if embedding_url else {}
    query_vec = await get_query_embedding(query, **kwargs)
    if query_vec is None:
        return None

    conn = db.connection()
    rows = conn.execute(
        text(
            "SELECT v.rowid, v.distance, a.content, a.tags, a.created_at, "
            "a.importance, a.emotion_tags, a.member_id, a.is_active "
            "FROM vec_archival v "
            "JOIN archival_memories a ON a.id = v.rowid "
            "WHERE v.embedding MATCH :qvec AND k = :k "
            "ORDER BY v.distance"
        ),
        {"qvec": _pack_vec(query_vec), "k": limit * 3},
    ).fetchall()

    now = datetime.datetime.utcnow()
    candidates = []
    for row in rows:
        row_id, distance, content, tags, created_at, importance, emotion_tags, mem_mid, is_active = row
        if is_active is not None and is_active == 0:
            continue
        if member_id is not None and mem_mid != member_id:
            continue
        similarity = max(0.0, 1.0 - distance / 2.0)
        if similarity <= 0.3:
            continue
        imp = importance if importance is not None else 0.5
        score = 0.7 * similarity + 0.3 * imp
        candidates.append({
            "id": row_id,
            "content": content,
            "tags": tags,
            "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
            "score": round(score, 4),
            "similarity": round(similarity, 4),
            "importance": imp,
            "emotion_tags": emotion_tags,
            "member_id": mem_mid,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    results = candidates[:limit]
    matched_ids = [r["id"] for r in results]

    # Boost importance for retrieved memories (+0.05, capped at 1.0)
    if matched_ids:
        for mem in db.query(ArchivalMemory).filter(ArchivalMemory.id.in_(matched_ids)):
            mem.last_accessed = now
            current_imp = mem.importance if mem.importance is not None else 0.5
            mem.importance = min(1.0, current_imp + 0.05)
        db.commit()

    return results


# ---------------------------------------------------------------------------
# Read-only similarity search (for deduplication — no retrieval boosting)
# ---------------------------------------------------------------------------

def find_similar_memories(
    db: Session,
    embedding: list[float],
    threshold: float = 0.85,
    limit: int = 5,
    member_id: str | None = None,
) -> list[dict]:
    """Find archival memories similar to a given embedding vector.

    Pure read-only — does NOT update last_accessed or importance.
    Used for deduplication checks before saving new memories.
    """
    conn = db.connection()
    over_fetch = limit * 3

    rows = conn.execute(
        text(
            "SELECT v.rowid, v.distance, a.content, a.tags, a.importance "
            "FROM vec_archival v "
            "JOIN archival_memories a ON a.id = v.rowid "
            "WHERE v.embedding MATCH :qvec AND k = :k "
            "ORDER BY v.distance"
        ),
        {"qvec": _pack_vec(embedding), "k": over_fetch},
    ).fetchall()

    results = []
    for row_id, distance, content, tags, importance in rows:
        similarity = max(0.0, 1.0 - distance / 2.0)
        if similarity < threshold:
            continue
        if member_id is not None:
            mem = db.query(ArchivalMemory).filter(
                ArchivalMemory.id == row_id,
                ArchivalMemory.member_id == member_id,
                ArchivalMemory.is_active == 1,
            ).first()
            if not mem:
                continue
        else:
            mem = db.query(ArchivalMemory).filter(
                ArchivalMemory.id == row_id,
                ArchivalMemory.is_active == 1,
            ).first()
            if not mem:
                continue
        results.append({
            "id": row_id,
            "content": content,
            "tags": tags,
            "similarity": round(similarity, 4),
            "importance": importance if importance is not None else 0.5,
        })
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Embedding update helper (for deduplication merges)
# ---------------------------------------------------------------------------

async def _update_memory_embedding(
    db: Session,
    memory_id: int,
    new_content: str,
    embedding_url: str | None = None,
):
    """Re-compute and update embedding after a memory's content changes."""
    kwargs = {"base_url": embedding_url} if embedding_url else {}
    embeddings = await get_embeddings([new_content], **kwargs)
    if not embeddings:
        logger.warning("Could not update embedding for memory %d (service unavailable)", memory_id)
        return

    new_embedding = embeddings[0]
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return

    mem.embedding = json.dumps(new_embedding)
    db.commit()

    try:
        insert_vec_embedding(db, memory_id, new_embedding)
    except Exception:
        logger.warning("Failed to update vec embedding for memory %d", memory_id)


# ---------------------------------------------------------------------------
# Deduplication-aware memory save
# ---------------------------------------------------------------------------

async def deduplicate_and_save(
    db: Session,
    content: str,
    tags: str = "",
    member_id: str | None = None,
    importance: float = 0.5,
    emotion_tags: str | None = None,
    embedding_url: str | None = None,
) -> dict:
    """Save a memory with deduplication and surprise-based importance.

    Checks new memory against existing ones by semantic similarity:
    - High similarity (>= 0.85): merge/skip (deduplication)
    - Moderate similarity (0.30-0.85): save with surprise-boosted importance
    - Low similarity (< 0.30) or no matches: save with neutral importance

    Surprise scoring (Titans-inspired): novel memories that relate to but differ
    from existing knowledge start with higher importance.
    """
    kwargs = {"base_url": embedding_url} if embedding_url else {}
    embeddings = await get_embeddings([content], **kwargs)
    if not embeddings:
        logger.info("Dedup: embedding service unavailable, saving without dedup check")
        mem = add_archival(db, content, tags, None, member_id, importance, emotion_tags)
        return {"action": "created", "memory_id": mem.id, "reason": "no_embedding_service"}

    new_embedding = embeddings[0]
    similar = find_similar_memories(db, new_embedding, threshold=0.0, limit=5, member_id=member_id)
    best_sim = similar[0]["similarity"] if similar else 0.0

    # --- Dedup check (>= 0.85): merge or skip ---
    if best_sim >= 0.85:
        best_match = similar[0]
        existing_id = best_match["id"]
        existing_content = best_match["content"]
        new_richer = len(content) > len(existing_content)

        if new_richer:
            revise_archival(db, existing_id, content,
                           reason=f"Dedup merge (sim={best_sim:.2f}): new version richer")
            existing_tags = best_match.get("tags", "")
            merged_tags = _merge_tags(existing_tags, tags)
            existing_mem = get_archival_by_id(db, existing_id)
            if existing_mem:
                existing_mem.tags = merged_tags
                db.commit()
            await _update_memory_embedding(db, existing_id, content, embedding_url=embedding_url)
            logger.info(
                "Dedup: merged into #%d (sim=%.2f, richer: %d>%d chars)",
                existing_id, best_sim, len(content), len(existing_content),
            )
            return {"action": "merged", "memory_id": existing_id, "similarity": best_sim}
        else:
            logger.info(
                "Dedup: skipped (sim=%.2f with #%d, existing richer: %d>=%d chars)",
                best_sim, existing_id, len(existing_content), len(content),
            )
            return {"action": "skipped", "memory_id": existing_id, "similarity": best_sim}

    # --- Surprise-based importance (Titans-inspired) ---
    if best_sim >= 0.30:
        surprise = (0.85 - best_sim) / 0.55
        importance = 0.5 + 0.2 * surprise
    elif similar:
        surprise = 1.0
        importance = 0.5
    else:
        surprise = 0.5
        importance = 0.5

    mem = add_archival(db, content, tags, new_embedding, member_id,
                       importance, emotion_tags, surprise=round(surprise, 4))
    logger.info(
        "Dedup: created #%d (best_sim=%.2f, surprise=%.2f, importance=%.2f)",
        mem.id, best_sim, surprise, importance,
    )
    return {
        "action": "created", "memory_id": mem.id,
        "surprise": round(surprise, 3), "importance": round(importance, 3),
    }


def _merge_tags(existing: str, new: str) -> str:
    """Merge two comma-separated tag strings, preserving order, deduplicating."""
    existing_tags = [t.strip() for t in existing.split(",") if t.strip()] if existing else []
    new_tags = [t.strip() for t in new.split(",") if t.strip()] if new else []
    seen = set(existing_tags)
    for tag in new_tags:
        if tag not in seen:
            existing_tags.append(tag)
            seen.add(tag)
    return ",".join(existing_tags)


# ---------------------------------------------------------------------------
# Unified retrieval: semantic with keyword fallback
# ---------------------------------------------------------------------------

async def retrieve_memories(
    db: Session,
    message: str,
    keywords: list[str],
    limit: int = 5,
    member_id: str | None = None,
    embedding_url: str | None = None,
) -> tuple[list[dict], str]:
    """Retrieve relevant archival memories, preferring semantic search.

    Returns (memories, method) where method is 'semantic' or 'keyword'.
    """
    results = await semantic_search(db, message, limit=limit, member_id=member_id,
                                    embedding_url=embedding_url)
    if results is not None:
        if results:
            return results, "semantic"
        keyword_results = search_archival_multi(db, keywords, limit=limit, member_id=member_id)
        if keyword_results:
            return keyword_results, "keyword"
        return [], "semantic"

    logger.info("Falling back to keyword search (embedding service unavailable)")
    return search_archival_multi(db, keywords, limit=limit, member_id=member_id), "keyword"


# ---------------------------------------------------------------------------
# Core memory blocks
# ---------------------------------------------------------------------------

def get_core_blocks(db: Session) -> list[dict]:
    """Get all core memory blocks."""
    blocks = db.query(MemoryBlock).order_by(MemoryBlock.label).all()
    return [
        {"label": b.label, "value": b.value, "updated_at": b.updated_at.isoformat()}
        for b in blocks
    ]


def get_block(db: Session, label: str) -> MemoryBlock | None:
    return db.query(MemoryBlock).filter(MemoryBlock.label == label).first()


def snapshot_block(db: Session, label: str, trigger: str = "api_update"):
    """Save a snapshot of the current block value before modification."""
    block = get_block(db, label)
    if block and block.value:
        snapshot = CoreBlockSnapshot(
            block_label=label,
            value=block.value,
            trigger=trigger,
        )
        db.add(snapshot)
        old = (
            db.query(CoreBlockSnapshot)
            .filter(CoreBlockSnapshot.block_label == label)
            .order_by(CoreBlockSnapshot.snapshot_at.desc())
            .offset(20)
            .all()
        )
        for s in old:
            db.delete(s)


def upsert_block(db: Session, label: str, value: str, trigger: str = "api_update") -> MemoryBlock:
    block = get_block(db, label)
    if block:
        snapshot_block(db, label, trigger=trigger)
        block.value = value
        block.updated_at = datetime.datetime.utcnow()
    else:
        block = MemoryBlock(label=label, value=value)
        db.add(block)
    db.commit()
    db.refresh(block)
    return block


def get_block_history(db: Session, label: str, limit: int = 20) -> list[dict]:
    """Get snapshot history for a core block."""
    snapshots = (
        db.query(CoreBlockSnapshot)
        .filter(CoreBlockSnapshot.block_label == label)
        .order_by(CoreBlockSnapshot.snapshot_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "block_label": s.block_label,
            "value_preview": s.value[:200] if s.value else "",
            "value_length": len(s.value) if s.value else 0,
            "trigger": s.trigger,
            "snapshot_at": s.snapshot_at.isoformat() if s.snapshot_at else "",
        }
        for s in snapshots
    ]


def restore_block(db: Session, label: str, snapshot_id: int) -> dict:
    """Restore a core block to a previous snapshot."""
    snapshot = (
        db.query(CoreBlockSnapshot)
        .filter(CoreBlockSnapshot.id == snapshot_id, CoreBlockSnapshot.block_label == label)
        .first()
    )
    if not snapshot:
        return {"restored": False, "error": f"Snapshot {snapshot_id} not found for block '{label}'"}

    snapshot_block(db, label, trigger="restore")
    block = get_block(db, label)
    if block:
        block.value = snapshot.value
        block.updated_at = datetime.datetime.utcnow()
        db.commit()
        return {"restored": True, "label": label, "restored_from": snapshot.snapshot_at.isoformat()}
    return {"restored": False, "error": f"Block '{label}' not found"}


def get_member_core_memory(db: Session, member_id: str) -> str:
    """Get all core memory blocks for a specific member, formatted for injection."""
    prefix = f"{member_id}:"
    blocks = (
        db.query(MemoryBlock)
        .filter(MemoryBlock.label.startswith(prefix))
        .order_by(MemoryBlock.label)
        .all()
    )
    if not blocks:
        return ""
    parts = []
    for b in blocks:
        short_label = b.label[len(prefix):]
        parts.append(f"<{short_label}>\n{b.value}\n</{short_label}>")
    return "\n\n".join(parts)


def delete_block(db: Session, label: str) -> bool:
    block = get_block(db, label)
    if block:
        db.delete(block)
        db.commit()
        return True
    return False


# ---------------------------------------------------------------------------
# Archival memory CRUD
# ---------------------------------------------------------------------------

def add_archival(
    db: Session,
    content: str,
    tags: str = "",
    embedding: list[float] | None = None,
    member_id: str | None = None,
    importance: float = 0.5,
    emotion_tags: str | None = None,
    surprise: float | None = None,
) -> ArchivalMemory:
    """Add an archival memory, optionally with a pre-computed embedding."""
    entry = ArchivalMemory(
        content=content,
        tags=tags,
        embedding=json.dumps(embedding) if embedding else None,
        member_id=member_id,
        importance=importance,
        emotion_tags=emotion_tags,
        surprise=surprise,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    if embedding:
        try:
            insert_vec_embedding(db, entry.id, embedding)
        except Exception:
            logger.warning("Failed to insert vec embedding for memory %d", entry.id)
    return entry


async def add_archival_with_embedding(
    db: Session,
    content: str,
    tags: str = "",
    member_id: str | None = None,
    importance: float = 0.5,
    emotion_tags: str | None = None,
    embedding_url: str | None = None,
) -> ArchivalMemory:
    """Add an archival memory and attempt to generate an embedding for it."""
    embedding = None
    kwargs = {"base_url": embedding_url} if embedding_url else {}
    embeddings = await get_embeddings([content], **kwargs)
    if embeddings:
        embedding = embeddings[0]
    else:
        logger.info("Saving memory without embedding (service unavailable)")

    return add_archival(db, content, tags, embedding, member_id, importance, emotion_tags)


def search_archival(db: Session, query: str, limit: int = 10, member_id: str | None = None) -> list[dict]:
    """Keyword search across archival memories."""
    # Escape SQL LIKE wildcards in user input to prevent enumeration
    escaped = query.replace("%", "\\%").replace("_", "\\_")
    q = f"%{escaped}%"
    filters = [
        ArchivalMemory.is_active != 0,
        (ArchivalMemory.content.ilike(q)) | (ArchivalMemory.tags.ilike(q)),
    ]
    if member_id is not None:
        filters.append(ArchivalMemory.member_id == member_id)
    results = (
        db.query(ArchivalMemory)
        .filter(*filters)
        .order_by(ArchivalMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "content": r.content,
            "tags": r.tags,
            "created_at": r.created_at.isoformat(),
        }
        for r in results
    ]


def list_archival(db: Session, limit: int = 50) -> list[dict]:
    results = (
        db.query(ArchivalMemory)
        .filter(ArchivalMemory.is_active != 0)
        .order_by(ArchivalMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "content": r.content,
            "tags": r.tags,
            "created_at": r.created_at.isoformat(),
        }
        for r in results
    ]


def search_archival_multi(db: Session, keywords: list[str], limit: int = 5, member_id: str | None = None) -> list[dict]:
    """Search archival memories using multiple keywords, return unique results."""
    seen_ids: set[int] = set()
    results: list[dict] = []
    for kw in keywords:
        if not kw.strip():
            continue
        for r in search_archival(db, kw.strip(), limit=limit, member_id=member_id):
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(r)
    return results[:limit]


# ---------------------------------------------------------------------------
# Memory curation
# ---------------------------------------------------------------------------

def get_archival_by_id(db: Session, memory_id: int) -> ArchivalMemory | None:
    """Fetch a single archival memory by ID."""
    return db.query(ArchivalMemory).filter(ArchivalMemory.id == memory_id).first()


def log_memory_edit(
    db: Session,
    memory_id: int,
    action: str,
    original_content: str | None = None,
    new_content: str | None = None,
    reason: str | None = None,
):
    """Write an entry to the memory_edits audit trail."""
    edit = MemoryEdit(
        memory_id=memory_id,
        action=action,
        original_content=original_content,
        new_content=new_content,
        reason=reason,
    )
    db.add(edit)
    db.commit()


def revise_archival(db: Session, memory_id: int, new_content: str, reason: str | None = None) -> dict:
    """Revise the content of an archival memory with audit trail."""
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return {"revised": False, "error": f"Memory {memory_id} not found"}
    if mem.is_active == 0:
        return {"revised": False, "error": f"Memory {memory_id} is forgotten. Restore it first."}

    original = mem.content
    log_memory_edit(db, memory_id, "revise", original_content=original, new_content=new_content, reason=reason)
    mem.content = new_content
    db.commit()
    return {"revised": True, "memory_id": memory_id, "original_preview": original[:100], "new_preview": new_content[:100]}


def deactivate_archival(db: Session, memory_id: int, reason: str | None = None) -> dict:
    """Soft-delete an archival memory (set is_active=0)."""
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return {"forgotten": False, "error": f"Memory {memory_id} not found"}
    if mem.is_active == 0:
        return {"forgotten": False, "error": f"Memory {memory_id} is already forgotten"}

    log_memory_edit(db, memory_id, "forget", original_content=mem.content, reason=reason)
    mem.is_active = 0
    db.commit()
    return {"forgotten": True, "memory_id": memory_id, "content_preview": mem.content[:100]}


def activate_archival(db: Session, memory_id: int) -> dict:
    """Restore a soft-deleted archival memory."""
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return {"restored": False, "error": f"Memory {memory_id} not found"}
    if mem.is_active != 0:
        return {"restored": False, "error": f"Memory {memory_id} is already active"}

    log_memory_edit(db, memory_id, "restore")
    mem.is_active = 1
    db.commit()
    return {"restored": True, "memory_id": memory_id, "content_preview": mem.content[:100]}


def protect_archival(db: Session, memory_id: int) -> dict:
    """Shield a memory from temporal decay. Boosts importance to 0.7 floor."""
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return {"protected": False, "error": f"Memory {memory_id} not found"}
    if mem.protected == 1:
        return {"protected": False, "error": f"Memory {memory_id} is already protected"}

    log_memory_edit(db, memory_id, "protect")
    mem.protected = 1
    if (mem.importance or 0.5) < 0.7:
        mem.importance = 0.7
    db.commit()
    return {"protected": True, "memory_id": memory_id, "importance": mem.importance, "content_preview": mem.content[:100]}


def release_archival(db: Session, memory_id: int) -> dict:
    """Remove decay protection from a memory."""
    mem = get_archival_by_id(db, memory_id)
    if not mem:
        return {"released": False, "error": f"Memory {memory_id} not found"}
    if mem.protected != 1:
        return {"released": False, "error": f"Memory {memory_id} is not currently protected"}

    log_memory_edit(db, memory_id, "release")
    mem.protected = 0
    db.commit()
    return {"released": True, "memory_id": memory_id, "content_preview": mem.content[:100]}


def search_archival_for_review(
    db: Session, query: str, limit: int = 10, show_forgotten: bool = False
) -> list[dict]:
    """Search archival memories with full metadata for curation review."""
    q = f"%{query}%"
    base = db.query(ArchivalMemory).filter(
        (ArchivalMemory.content.ilike(q)) | (ArchivalMemory.tags.ilike(q))
    )
    if not show_forgotten:
        base = base.filter(ArchivalMemory.is_active != 0)
    results = base.order_by(ArchivalMemory.created_at.desc()).limit(limit).all()

    now = datetime.datetime.utcnow()
    return [
        {
            "id": r.id,
            "content": r.content,
            "tags": r.tags,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "importance": r.importance if r.importance is not None else 0.5,
            "age_days": round((now - r.created_at).total_seconds() / 86400, 1) if r.created_at else 0,
            "is_active": r.is_active if r.is_active is not None else 1,
            "protected": r.protected if r.protected is not None else 0,
            "emotion_tags": r.emotion_tags or "",
        }
        for r in results
    ]


def get_memory_edits(db: Session, memory_id: int | None = None, limit: int = 50) -> list[dict]:
    """Retrieve audit trail entries, optionally filtered by memory_id."""
    base = db.query(MemoryEdit)
    if memory_id is not None:
        base = base.filter(MemoryEdit.memory_id == memory_id)
    results = base.order_by(MemoryEdit.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "memory_id": r.memory_id,
            "action": r.action,
            "original_content": r.original_content,
            "new_content": r.new_content,
            "reason": r.reason,
            "timestamp": r.timestamp.isoformat() if r.timestamp else "",
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Session summaries
# ---------------------------------------------------------------------------

def get_recent_summaries(db: Session, limit: int = 5) -> list[dict]:
    """Get the most recent session summaries in chronological order."""
    summaries = (
        db.query(SessionSummary)
        .filter(SessionSummary.period_type == "session")
        .order_by(SessionSummary.created_at.desc())
        .limit(limit)
        .all()
    )
    summaries.reverse()
    return [
        {
            "id": s.id,
            "summary": s.summary,
            "key_topics": s.key_topics,
            "emotional_arc": s.emotional_arc,
            "message_count": s.message_count,
            "period_start": s.period_start.isoformat() if s.period_start else "",
            "period_end": s.period_end.isoformat() if s.period_end else "",
            "created_at": s.created_at.isoformat() if s.created_at else "",
        }
        for s in summaries
    ]


def get_recent_agent_summaries(db: Session, limit: int = 3) -> list[dict]:
    """Get the most recent agent session summaries from archival memory."""
    memories = (
        db.query(ArchivalMemory)
        .filter(ArchivalMemory.tags.contains("inner_life,agent,session"))
        .order_by(ArchivalMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    memories.reverse()
    return [
        {
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else "",
        }
        for m in memories
    ]


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _char_limit_for_score(score: float | None) -> int:
    """Return per-memory character limit based on cosine similarity score."""
    if score is None:
        return 1000
    if score >= 0.7:
        return 2000
    if score >= 0.5:
        return 1000
    return 500


ARCHIVAL_BUDGET = 8000


def build_archival_context(memories: list[dict]) -> str:
    """Format retrieved archival memories for injection into the prompt."""
    if not memories:
        return ""

    parts = []
    remaining = ARCHIVAL_BUDGET
    for m in memories:
        if remaining <= 0:
            break
        score = m.get("score")
        limit = min(_char_limit_for_score(score), remaining)
        content = m["content"][:limit]
        parts.append(f"- {content}")
        remaining -= len(content)

    return (
        "### Retrieved Memories\n"
        "The following archival memories may be relevant to this conversation:\n\n"
        + "\n\n".join(parts)
    )


def build_memory_context(db: Session) -> str:
    """Build the memory context string to prepend to conversations."""
    blocks = db.query(MemoryBlock).order_by(MemoryBlock.label).all()
    if not blocks:
        return ""

    sections = []
    for block in blocks:
        # Sanitize label to prevent tag injection (alphanumeric + underscore only)
        safe_label = "".join(c for c in block.label if c.isalnum() or c == "_") or "block"
        # Escape value to prevent closing-tag injection
        safe_value = block.value.replace(f"</{safe_label}>", f"&lt;/{safe_label}&gt;")
        sections.append(f"<{safe_label}>\n{safe_value}\n</{safe_label}>")

    return (
        "### Core Memories\n"
        "The following are your persistent core memories. "
        "They define who you are, who you're talking to, and important context.\n\n"
        + "\n\n".join(sections)
    )
