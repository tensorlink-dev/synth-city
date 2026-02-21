"""
Basilica GPU client — thin wrapper around the ``basilica-sdk`` package that
restricts rentals to a configurable set of cheap GPU offerings.

The official SDK talks to the Basilica secure-cloud marketplace (Hyperstack,
Verda, etc.).  This module adds:

* **Price filtering** — only offerings ≤ ``BASILICA_MAX_HOURLY_RATE`` are shown.
* **GPU-type allowlist** — only GPU types in ``BASILICA_ALLOWED_GPU_TYPES``.
* **Convenience helpers** — ``rent_cheapest()`` picks the cheapest available GPU.
"""

from __future__ import annotations

import logging
from typing import Any

from basilica import BasilicaClient as _SdkClient
from basilica import GpuOffering, SecureCloudRentalResponse

from config import (
    BASILICA_ALLOWED_GPU_TYPES,
    BASILICA_API_TOKEN,
    BASILICA_API_URL,
    BASILICA_MAX_HOURLY_RATE,
)

logger = logging.getLogger(__name__)


class BasilicaGPUClient:
    """Budget-aware wrapper around the Basilica secure-cloud GPU API.

    Filters offerings to only those matching *allowed_gpu_types* and costing
    at most *max_hourly_rate* USD per GPU-hour.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_hourly_rate: float | None = None,
        allowed_gpu_types: list[str] | None = None,
    ) -> None:
        token = api_key or BASILICA_API_TOKEN
        if not token:
            raise RuntimeError(
                "BASILICA_API_TOKEN not set. Add it to .env to use Basilica GPU cloud."
            )
        self._client = _SdkClient(
            api_key=token,
            base_url=base_url or BASILICA_API_URL,
        )
        if max_hourly_rate is not None:
            self.max_hourly_rate = max_hourly_rate
        else:
            self.max_hourly_rate = BASILICA_MAX_HOURLY_RATE
        self.allowed_gpu_types = [
            t.upper() for t in (allowed_gpu_types or BASILICA_ALLOWED_GPU_TYPES)
        ]

    # ------------------------------------------------------------------
    # GPU discovery
    # ------------------------------------------------------------------

    def list_cheap_gpus(self) -> list[GpuOffering]:
        """Return GPU offerings filtered by price cap and GPU-type allowlist.

        Results are sorted by ``hourly_rate`` ascending (cheapest first).
        """
        all_offerings = self._client.list_secure_cloud_gpus()
        filtered = [
            o for o in all_offerings
            if float(o.hourly_rate) <= self.max_hourly_rate
            and any(allowed.upper() in o.gpu_type.upper() for allowed in self.allowed_gpu_types)
        ]
        filtered.sort(key=lambda o: float(o.hourly_rate))
        return filtered

    # ------------------------------------------------------------------
    # Rental lifecycle
    # ------------------------------------------------------------------

    def rent_gpu(
        self,
        offering_id: str,
        ssh_public_key_id: str | None = None,
    ) -> SecureCloudRentalResponse:
        """Start a secure-cloud GPU rental for a specific offering."""
        resp = self._client.start_secure_cloud_rental(
            offering_id=offering_id,
            ssh_public_key_id=ssh_public_key_id,
        )
        logger.info(
            "Started rental %s — %s @ $%s/hr (provider=%s)",
            resp.rental_id, resp.provider, resp.hourly_cost, resp.provider,
        )
        return resp

    def rent_cheapest(
        self,
        ssh_public_key_id: str | None = None,
    ) -> SecureCloudRentalResponse:
        """Rent the cheapest currently-available GPU within the budget.

        Raises ``RuntimeError`` if no offerings match.
        """
        offerings = self.list_cheap_gpus()
        if not offerings:
            raise RuntimeError(
                f"No GPU offerings available within ${self.max_hourly_rate}/hr "
                f"for types {self.allowed_gpu_types}"
            )
        cheapest = offerings[0]
        logger.info(
            "Auto-selecting cheapest offering: %s %s @ $%s/hr (id=%s)",
            cheapest.gpu_type,
            "(Spot)" if cheapest.is_spot else "",
            cheapest.hourly_rate,
            cheapest.id,
        )
        return self.rent_gpu(cheapest.id, ssh_public_key_id=ssh_public_key_id)

    def stop_rental(self, rental_id: str) -> dict[str, Any]:
        """Stop a secure-cloud rental and return cost summary."""
        resp = self._client.stop_secure_cloud_rental(rental_id)
        logger.info(
            "Stopped rental %s — ran %.2fh, total $%.4f",
            resp.rental_id, resp.duration_hours, resp.total_cost,
        )
        return {
            "rental_id": resp.rental_id,
            "status": resp.status,
            "duration_hours": resp.duration_hours,
            "total_cost": resp.total_cost,
        }

    def list_active_rentals(self) -> list[dict[str, Any]]:
        """List all secure-cloud rentals for this account."""
        resp = self._client.list_secure_cloud_rentals()
        return [
            {
                "rental_id": getattr(r, "rental_id", None),
                "status": getattr(r, "status", None),
                "provider": getattr(r, "provider", None),
                "hourly_cost": getattr(r, "hourly_cost", None),
                "ip_address": getattr(r, "ip_address", None),
                "ssh_command": getattr(r, "ssh_command", None),
            }
            for r in (resp.rentals if hasattr(resp, "rentals") else [])
        ]

    def get_rental_status(self, rental_id: str) -> dict[str, Any]:
        """Get detailed status for a single rental (includes SSH info)."""
        r = self._client.get_rental(rental_id)
        return {
            "rental_id": r.rental_id,
            "status": getattr(r.status, "state", str(r.status)),
            "ssh_credentials": {
                "host": r.ssh_credentials.host,
                "port": r.ssh_credentials.port,
                "user": r.ssh_credentials.user,
            } if r.ssh_credentials else None,
            "created_at": str(r.created_at) if r.created_at else None,
        }

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """Return account balance information."""
        return self._client.get_balance()

    def ensure_ssh_key(self, name: str = "synth-city", public_key_path: str | None = None) -> str:
        """Register an SSH key if one is not already registered.

        Returns the SSH key ID.
        """
        existing = self._client.get_ssh_key()
        if existing:
            logger.info("SSH key already registered: %s (id=%s)", existing.name, existing.id)
            return existing.id

        resp = self._client.register_ssh_key(
            name=name,
            public_key_path=public_key_path or "~/.ssh/id_ed25519.pub",
        )
        logger.info("Registered SSH key: %s (id=%s)", resp.name, resp.id)
        return resp.id
