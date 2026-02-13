"""
synth-city bridge client — convenience wrapper for calling the bridge HTTP API.

This module can be used standalone or imported by OpenClaw workspace skills
to interact with a running synth-city bridge server.

Usage::

    from integrations.openclaw.client import SynthCityClient

    client = SynthCityClient()  # default: http://127.0.0.1:8377
    print(client.list_blocks())
    print(client.pipeline_status())
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class SynthCityClient:
    """HTTP client for the synth-city bridge server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8377", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = httpx.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = httpx.post(f"{self.base_url}{path}", json=body or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

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
