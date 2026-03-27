# Security Guide for Auditors

This document is for anyone — human or AI — performing a security review of amarin-memory. It describes the threat model, trust boundaries, known limitations, and where to focus your attention.

## Threat Model

amarin-memory is a **library**, not a service. It runs inside the caller's process, using a local SQLite database. There is no network-facing API, no authentication layer, and no server component. The security surface depends entirely on how the library is deployed.

### Deployment Scenarios (in order of risk)

1. **Single-agent, single-user** — One agent, one database, one operator. Lowest risk. The operator controls all inputs.

2. **Multi-agent, shared database** — Multiple agents share one SQLite file, isolated by `member_id`. This is the primary multi-tenant scenario. Trust boundary: agents should not be able to read or mutate each other's memories.

3. **Agent with untrusted user input** — An agent (e.g., via OpenClaw) stores memories derived from user conversation. Trust boundary: user-controlled content should not enable code execution, memory disclosure, or prompt injection.

4. **Shared hosting / multi-operator** — Multiple operators on the same machine, each running agents. Trust boundary: environment variables, file permissions, process isolation. This library does not provide OS-level isolation.

## Trust Boundaries

| Boundary | What crosses it | Risk |
|----------|----------------|------|
| User input -> memory content | Conversation text stored as memories | Prompt injection, shell injection (via CLI) |
| member_id isolation | Queries filtered by tenant | Cross-tenant disclosure if filters are bypassed |
| Embedding service URL | Memory content sent to external HTTP endpoint | SSRF if URL is attacker-controlled |
| Core blocks -> LLM prompt | Block values injected into system prompts | Prompt injection via crafted block content |
| CLI arguments | Shell commands with user-derived content | Command injection if not properly quoted |

## Known Limitations

### KNN search is global, filtered post-query
sqlite-vec does not support filtered KNN. Semantic search queries the global vector index and filters by `member_id` afterward. In a shared database with many tenants, other tenants' results can crowd out same-tenant matches (the query over-fetches by 3x to compensate, but this is not a guarantee).

**Mitigation:** For strict tenant isolation, use separate database files per agent rather than shared-database multi-tenancy.

### Low-level functions commit internally
Functions like `add_archival`, `revise_archival`, `log_memory_edit` call `db.commit()` internally. Callers cannot compose them into atomic transactions. A failure mid-operation can leave partial state.

**Mitigation:** Use the high-level `MemoryEngine` API for most operations. If you need transactional control, you'll need to modify the functions to accept a `commit=False` parameter (not yet implemented).

### No authentication or authorization
The library has no concept of "who is calling." The `member_id` parameter is caller-asserted, not verified. Any code that can import the library can access any tenant's data.

**Mitigation:** This is by design for a library. Authorization belongs in the service layer that wraps this library, not in the library itself.

## What to Look For

If you're auditing this codebase, focus on:

1. **SQL injection** — All queries use SQLAlchemy ORM or parameterized `text()` queries. Check that no raw string formatting reaches SQL execution.

2. **Cross-tenant data leakage** — Every query path that accepts `member_id` should filter consistently. Watch for fallback paths (keyword search, list operations) that might skip the filter.

3. **Prompt injection via stored content** — `build_memory_context()` and `build_archival_context()` format stored content for LLM consumption. Check that stored content cannot break out of the intended format.

4. **Shell injection via CLI** — The OpenClaw skill instructs agents to run CLI commands. Check that user-derived content cannot escape argument boundaries. Prefer stdin over command-line arguments for untrusted content.

5. **Embedding service trust** — The library sends memory content to the configured embedding URL via HTTP POST. Check for SSRF vectors, response validation, and plaintext exposure.

6. **Dependency vulnerabilities** — Check `pyproject.toml` version pins against known CVEs.

## Reporting

If you find a vulnerability, please open an issue at:
https://github.com/flaggdavid-source/amarin-memory/issues

Or contact the maintainer directly at flaggdavid84@gmail.com.

For critical vulnerabilities (RCE, data exfiltration), please use email rather than public issues.
