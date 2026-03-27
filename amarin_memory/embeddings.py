"""Async embedding client for the amarin-memory package."""

import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("amarin_memory.embeddings")

DEFAULT_EMBEDDING_URL = "http://localhost:8200"
DEFAULT_TIMEOUT = 15.0


def _validate_url(url: str) -> str:
    """Validate embedding service URL to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Embedding URL must use http or https, got: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError(f"Invalid embedding URL: {url}")
    return url


async def get_embeddings(
    texts: list[str],
    base_url: str = DEFAULT_EMBEDDING_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[list[list[float]]]:
    """Get document embeddings from the embedding service.

    Returns None if the service is unavailable (caller should fall back).
    """
    try:
        _validate_url(base_url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/embed",
                json={"texts": texts},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
    except Exception as e:
        logger.warning("Embedding service unavailable: %s", e)
        return None


async def get_query_embedding(
    query: str,
    base_url: str = DEFAULT_EMBEDDING_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[list[float]]:
    """Get a query embedding (uses search_query prefix for nomic).

    Returns None if the service is unavailable.
    """
    try:
        _validate_url(base_url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/query-embed",
                json={"query": query},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
    except Exception as e:
        logger.warning("Embedding service unavailable: %s", e)
        return None
