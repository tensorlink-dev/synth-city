"""
synth-city bridge client — convenience wrapper for calling the bridge HTTP API.

This module can be used standalone or imported by OpenClaw workspace skills
to interact with a running synth-city bridge server.

Usage::

    from integrations.openclaw.client import SynthCityClient

    client = SynthCityClient()  # default: http://127.0.0.1:8377
    print(client.list_blocks())
    print(client.pipeline_status())

The bridge URL can be configured via the ``SYNTH_BRIDGE_URL`` environment
variable or by passing ``base_url`` to the constructor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BRIDGE_URL = os.getenv("SYNTH_BRIDGE_URL", "http://127.0.0.1:8377")
_DEFAULT_TIMEOUT = float(os.getenv("SYNTH_BRIDGE_TIMEOUT", "300"))

# Retry settings for transient failures
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds; doubles on each retry
_RETRYABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


class BridgeConnectionError(Exception):
    """Raised when the client cannot reach the bridge server."""


class SynthCityClient:
    """HTTP client for the synth-city bridge server."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BRIDGE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = _MAX_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an HTTP request with automatic retry on transient failures."""
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                if method == "GET":
                    resp = httpx.get(url, params=params, timeout=self.timeout)
                else:
                    resp = httpx.post(url, json=body or {}, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self.retries:
                    delay = _RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        "Bridge request %s %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        method, path, attempt, self.retries, exc, delay,
                    )
                    time.sleep(delay)
            except httpx.HTTPStatusError as exc:
                # Don't retry 4xx / 5xx — the server responded
                try:
                    return exc.response.json()
                except (json.JSONDecodeError, ValueError):
                    raise BridgeConnectionError(
                        f"Bridge returned HTTP {exc.response.status_code} for {method} {path}"
                    ) from exc

        raise BridgeConnectionError(
            f"Bridge unreachable at {url} after {self.retries} attempts. "
            f"Is it running? Start with: python main.py bridge\n"
            f"Last error: {last_exc}"
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("POST", path, body=body)

    # ---- health
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    # ---- pipeline
    def pipeline_run(
        self,
        channel: str = "default",
        retries: int = 5,
        temperature: float = 0.1,
        publish: bool = False,
    ) -> dict[str, Any]:
        return self._post("/pipeline/run", {
            "channel": channel,
            "retries": retries,
            "temperature": temperature,
            "publish": publish,
        })

    def pipeline_status(self) -> dict[str, Any]:
        return self._get("/pipeline/status")

    # ---- components
    def list_blocks(self) -> dict[str, Any]:
        return self._get("/components/blocks")

    def list_heads(self) -> dict[str, Any]:
        return self._get("/components/heads")

    def list_presets(self) -> dict[str, Any]:
        return self._get("/components/presets")

    # ---- experiments
    def create_experiment(
        self,
        blocks: list[str],
        head: str = "GBMHead",
        d_model: int = 32,
        horizon: int = 12,
        n_paths: int = 100,
        lr: float = 0.001,
    ) -> dict[str, Any]:
        return self._post("/experiment/create", {
            "blocks": blocks,
            "head": head,
            "d_model": d_model,
            "horizon": horizon,
            "n_paths": n_paths,
            "lr": lr,
        })

    def run_experiment(
        self,
        experiment: dict[str, Any],
        epochs: int = 1,
        name: str = "",
    ) -> dict[str, Any]:
        return self._post("/experiment/run", {
            "experiment": experiment,
            "epochs": epochs,
            "name": name,
        })

    def validate_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        return self._post("/experiment/validate", {"experiment": experiment})

    def compare_results(self) -> dict[str, Any]:
        return self._get("/experiment/compare")

    def session_summary(self) -> dict[str, Any]:
        return self._get("/session/summary")

    def clear_session(self) -> dict[str, Any]:
        return self._post("/session/clear")

    # ---- market data
    def get_price(self, asset: str) -> dict[str, Any]:
        return self._get(f"/market/price/{asset}")

    def get_history(self, asset: str, days: int = 30) -> dict[str, Any]:
        return self._get(f"/market/history/{asset}", params={"days": days})
