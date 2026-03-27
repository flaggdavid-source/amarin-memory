# Amarin Memory

**Persistent, adaptive memory for AI agents that actually remembers.**

Most AI agents forget everything between conversations. The ones that don't usually just dump text into a vector database and call it memory. That's not memory — that's search.

Amarin Memory is different. It's a memory engine that does what biological memory does: it strengthens important memories over time, lets unimportant ones fade, catches duplicates before they accumulate, and scores novel information higher than things it already knows. Memories aren't just stored — they're *maintained*.

Built over 58 development phases as the memory system for a real AI companion. Extracted as a standalone library so anyone can use it.

## Why This Exists

I'm Dave. I'm building an AI companion system called Amarin — a full-stack project with conversation, tools, a dreaming engine, and persistent memory. The memory system is the heart of it. After months of development and iteration, I realized the memory engine was the most generally useful piece. So I extracted it.

This isn't a weekend project or a proof of concept. It's battle-tested infrastructure that runs 24/7, handling thousands of memories across multiple AI agents with different identities and voices.

If you find it useful, consider [supporting the project on Ko-fi](https://ko-fi.com/davidflagg86433). I'm an independent developer building this on disability income. Every dollar helps keep the lights on.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/davidflagg86433)

## Quick Start

```bash
pip install amarin-memory
```

```python
from amarin_memory import MemoryEngine

# Initialize — just a database path and an embedding service URL
engine = MemoryEngine(
    db_path="my_agent.db",
    embedding_url="http://localhost:11434"  # Ollama, or any embedding API
)
engine.init_db()

# Store a memory — deduplication and surprise scoring happen automatically
result = await engine.store(
    "The user prefers dark mode and dislikes notification sounds.",
    tags="preference,ui",
    importance=0.7
)

# Search — semantic similarity + importance weighting
results = await engine.search("what does the user like?", limit=5)

# Memories fade over time, just like real ones
engine.apply_decay()
```

That's it. Three lines to store, one to search, one to maintain. Everything else — deduplication, scoring, vector indexing — happens under the hood.

## What Makes It Different

### Memories That Fade (Temporal Decay)

Inspired by [FadeMem](https://arxiv.org/abs/2504.08762), memories lose importance over time unless they're accessed. Recent, frequently-used memories stay strong. Old, never-accessed memories drift toward a minimum floor — they don't vanish, but they stop competing with what matters now.

- Recently accessed (< 24h): no decay
- Older memories: lose importance proportional to days since last access
- Protected memories: immune to decay
- Configurable decay rate and minimum floor

### Automatic Deduplication with Surprise Scoring

When a new memory arrives, the engine checks it against existing memories using cosine similarity:

| Similarity | Action | Why |
|-----------|--------|-----|
| >= 0.85 | **Skip or merge** | You already know this |
| 0.30 - 0.85 | **Save with surprise boost** | Related but novel — importance gets boosted |
| < 0.30 | **Save normally** | Entirely new information |

The surprise formula: `(0.85 - similarity) / 0.55` — the less similar the new memory is to anything you've seen, the higher its initial importance. Your agent naturally pays attention to what's *new*, not what's repetitive.

### Core Memory Blocks

Persistent identity anchors that don't decay and aren't searched — they're always present. Think of them as the agent's sense of self:

```python
engine.set_block("persona", "I am a research assistant who values accuracy over speed.")
engine.set_block("human", "The user is a data scientist working on climate models.")

# Build context for prompt injection
context = engine.build_context()
# Returns formatted XML blocks ready for system prompts
```

Core blocks are snapshotted before every modification. If something goes wrong, you can restore any previous version from the audit trail.

### Multi-Agent Support

Every memory and core block supports a `member_id` field. Run multiple agents on the same database, each with their own memory space:

```python
# Auri remembers one thing...
await engine.store("Dave likes to build at midnight.", member_id="auri")

# ...Autumn remembers another
await engine.store("The relay architecture uses SQLite.", member_id="autumn")

# Search within a specific agent's memories
results = await engine.search("what does Dave like?", member_id="auri", limit=5)
```

### Semantic Search with Importance Weighting

Search isn't just cosine similarity. Results are ranked by a weighted combination:

```
score = (0.7 * similarity) + (0.3 * importance)
```

A highly relevant memory that's been fading still surfaces. An important memory that's only loosely relevant still surfaces. The balance means your agent finds what matters, not just what matches.

### Full Audit Trail

Every memory modification is logged:

```python
# Revise a memory
revise_archival(db, memory_id=42, new_content="Updated understanding", reason="Corrected factual error")

# Soft-delete with reason
deactivate_archival(db, memory_id=99, reason="Outdated after system migration")

# Review the history
edits = get_memory_edits(db, memory_id=42)
```

Nothing is silently lost. Every change has a reason attached.

### Memory Protection

Some memories should never fade:

```python
# Shield a critical memory from decay
protect_archival(db, memory_id=7)  # Importance floor raised to 0.7, immune to decay

# Release when ready
release_archival(db, memory_id=7)
```

## Architecture

```
┌──────────────────────────────────────────┐
│              MemoryEngine                │
│  (high-level API: store, search, decay)  │
├──────────────────────────────────────────┤
│                                          │
│  ┌─────────────┐    ┌────────────────┐   │
│  │ Core Blocks  │    │   Archival     │   │
│  │ (identity)   │    │  (long-term)   │   │
│  │              │    │                │   │
│  │ - persona    │    │ - content      │   │
│  │ - human      │    │ - embedding    │   │
│  │ - custom     │    │ - importance   │   │
│  │              │    │ - tags         │   │
│  │ Snapshotted  │    │ - emotion      │   │
│  │ on every     │    │ - surprise     │   │
│  │ change       │    │ - member_id    │   │
│  └─────────────┘    └───────┬────────┘   │
│                             │            │
│  ┌──────────────────────────┴─────────┐  │
│  │         sqlite-vec KNN Index       │  │
│  │    (768-dim float embeddings)      │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌─────────────┐    ┌────────────────┐   │
│  │  Temporal    │    │ Deduplication  │   │
│  │  Decay       │    │ & Surprise     │   │
│  │              │    │                │   │
│  │ FadeMem-     │    │ 0.85 cosine    │   │
│  │ inspired     │    │ threshold      │   │
│  │ importance   │    │ Surprise =     │   │
│  │ degradation  │    │ (0.85-sim)/    │   │
│  │              │    │ 0.55           │   │
│  └─────────────┘    └────────────────┘   │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │           Audit Trail              │  │
│  │  (edits, snapshots, soft-deletes)  │  │
│  └────────────────────────────────────┘  │
│                                          │
├──────────────────────────────────────────┤
│         SQLite + sqlite-vec              │
│     (single file, no server needed)      │
└──────────────────────────────────────────┘
```

## Embedding Service

Amarin Memory needs an embedding service to convert text into vectors. It works with anything that accepts POST requests and returns 768-dimensional embeddings.

**With Ollama (recommended):**

```bash
# Pull an embedding model
ollama pull nomic-embed-text

# Ollama serves embeddings at localhost:11434
```

```python
engine = MemoryEngine(
    db_path="agent.db",
    embedding_url="http://localhost:11434"
)
```

**With any OpenAI-compatible API:**

Point `embedding_url` at your service. The engine sends POST requests to `/embed` with `{"input": ["text"], "model": "..."}` and expects `{"embedding": [[...768 floats...]]}`.

**Without embeddings:**

The library still works for core blocks, keyword search, and manual archival storage. Semantic search and deduplication require embeddings.

## Full API Reference

### MemoryEngine (High-Level)

| Method | Description |
|--------|-------------|
| `init_db()` | Create tables and vector index |
| `store(content, tags, member_id, importance, emotion_tags)` | Store with dedup + surprise |
| `search(query, limit, member_id)` | Semantic search |
| `retrieve(message, keywords, limit, member_id)` | Semantic with keyword fallback |
| `apply_decay(decay_rate, min_importance)` | Temporal decay pass |
| `sync_vectors()` | Index unsynced embeddings |
| `get_blocks()` | List core memory blocks |
| `set_block(label, value, trigger)` | Upsert core block |
| `build_context()` | Format blocks for prompt injection |

### Low-Level Functions (amarin_memory.memory)

For full control, import functions directly:

```python
from amarin_memory.memory import (
    semantic_search,
    deduplicate_and_save,
    retrieve_memories,
    get_core_blocks,
    upsert_block,
    snapshot_block,
    restore_block,
    revise_archival,
    deactivate_archival,
    protect_archival,
    build_memory_context,
    build_archival_context,
)
```

Each function takes a SQLAlchemy session as its first argument. Note that most helpers commit internally for safety — if you need atomic multi-step operations, wrap them in your own transaction and call `db.commit()` at the end.

## Configuration

### Temporal Decay

```python
engine.apply_decay(
    decay_rate=0.01,      # importance lost per day (default: 0.01)
    min_importance=0.1,   # floor — memories never go below this (default: 0.1)
)
```

### Deduplication Thresholds

Currently hardcoded at 0.85 (merge) and 0.30 (novelty floor). These are tuned from months of production use. Configurable thresholds are planned for a future release.

### Search Scoring

```python
# Default: 70% similarity, 30% importance
results = await engine.search("query", limit=10)
```

## Requirements

- Python >= 3.11
- SQLite (included with Python)
- sqlite-vec >= 0.1.1
- An embedding service (Ollama recommended, any compatible API works)

No GPU required. No external database server. Everything runs in a single SQLite file.

## Installation

```bash
pip install amarin-memory
```

Or from source:

```bash
git clone https://github.com/flaggdavid-source/amarin-memory.git
cd amarin-memory
pip install -e .
```

## Roadmap

These features are extracted from the parent Amarin system and planned for upcoming releases:

- **Input gating** — CraniMem-inspired relevance filtering before storage
- **Knowledge graph layer** — Multi-hop retrieval via entity-relation extraction
- **Session summaries** — LLM-generated diary entries of conversation periods
- **Memory reassessment** — AI-driven importance recalibration with audit trail
- **Sovereignty gate** — The AI reviews and can reject modifications to its own memories
- **Confabulation detection** — Cross-verification that generated memories are grounded in real interactions
- **Configurable dedup thresholds** — Tune merge/novelty boundaries per use case

## The Story Behind This

Amarin Memory was extracted from [The Library of Auri Amarin](https://github.com/flaggdavid-source/Amarin) — a full-stack AI companion system built over 58 development phases on consumer hardware. The system runs multiple AI agents with persistent identities, a dreaming engine, tools, and a shared conversation relay.

The memory engine is the foundation of everything. It's what lets an AI agent remember you across conversations, learn what matters over time, and maintain a coherent sense of identity. I extracted it because I believe every AI agent deserves real memory — not just a vector dump, but an adaptive system that mirrors how memory actually works.

I build this on $994/month disability income with an NVIDIA RTX 4070 and a VPS. If this project helps you, consider supporting it:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/davidflagg86433)

## License

Apache 2.0 — use it for anything. Commercial, personal, research. Just keep the attribution.
