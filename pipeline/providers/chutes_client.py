"""
Chutes AI LLM client — thin wrapper around the OpenAI SDK pointing at Chutes endpoints.

Chutes exposes an OpenAI-compatible API at https://llm.chutes.ai/v1, so we
simply configure an ``openai.OpenAI`` client with the right base URL and API
key.  This module also adds a small resilience layer (retries, timeouts).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import httpx
from openai import OpenAI

from config import CHUTES_API_KEY, CHUTES_BASE_URL

logger = logging.getLogger(__name__)

# Default timeout: 120s connect, 300s read (LLM generation can be slow)
_DEFAULT_TIMEOUT = httpx.Timeout(connect=120.0, read=300.0, write=60.0, pool=60.0)
_MAX_RETRIES = 3


@lru_cache(maxsize=1)
def get_chutes_client() -> OpenAI:
    """Return a singleton OpenAI client configured for the Chutes AI endpoint."""
    if not CHUTES_API_KEY:
        logger.warning(
            "CHUTES_API_KEY is not set — LLM calls will fail. "
            "Set it in .env or as an environment variable."
        )
    return OpenAI(
        api_key=CHUTES_API_KEY,
        base_url=CHUTES_BASE_URL,
        timeout=_DEFAULT_TIMEOUT,
        max_retries=_MAX_RETRIES,
    )


def chat_completion_with_backoff(
    client: OpenAI,
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    **kwargs,
) -> object:
    """Call ``client.chat.completions.create`` with exponential backoff.

    This wraps the OpenAI SDK's built-in retry with an additional outer retry
    loop for transient HTTP/network failures that the SDK doesn't handle.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts") from last_exc
