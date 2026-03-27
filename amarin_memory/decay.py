"""Temporal decay for archival memory importance scores (FadeMem-inspired)."""

import datetime
import logging

from sqlalchemy.orm import Session

from amarin_memory.models import ArchivalMemory

logger = logging.getLogger("amarin_memory.decay")


def apply_temporal_decay(
    db: Session,
    decay_rate: float = 0.01,
    min_importance: float = 0.1,
):
    """Apply temporal decay to archival memory importance scores.

    FadeMem-inspired: memories that haven't been accessed recently decay slowly.
    Memories accessed within the last 24 hours are not decayed.
    Protected memories are shielded from decay.

    Args:
        decay_rate: How much importance to subtract per day since last access.
        min_importance: Floor value — memories never decay below this.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=24)

    # Decay memories with last_accessed before cutoff (skip protected)
    memories = (
        db.query(ArchivalMemory)
        .filter(
            ArchivalMemory.importance > min_importance,
            ArchivalMemory.last_accessed.isnot(None),
            ArchivalMemory.last_accessed < cutoff,
            ArchivalMemory.protected != 1,
        )
        .all()
    )

    count = 0
    for mem in memories:
        days_since = (now - mem.last_accessed).total_seconds() / 86400.0
        decay = decay_rate * days_since
        new_importance = max(min_importance, mem.importance - decay)
        if new_importance < mem.importance:
            mem.importance = round(new_importance, 4)
            count += 1

    if count:
        db.commit()
        logger.info("Applied temporal decay to %d memories", count)

    # Also decay memories that have never been accessed (last_accessed is None)
    # These slowly lose importance from their initial value (half rate)
    unaccessed = (
        db.query(ArchivalMemory)
        .filter(
            ArchivalMemory.importance > min_importance,
            ArchivalMemory.last_accessed.is_(None),
            ArchivalMemory.protected != 1,
        )
        .all()
    )
    count2 = 0
    for mem in unaccessed:
        if mem.created_at:
            days_since = (now - mem.created_at).total_seconds() / 86400.0
            decay = decay_rate * 0.5 * days_since
            new_importance = max(min_importance, (mem.importance or 0.5) - decay)
            if new_importance < (mem.importance or 0.5):
                mem.importance = round(new_importance, 4)
                count2 += 1
    if count2:
        db.commit()
        logger.info("Applied decay to %d never-accessed memories", count2)
