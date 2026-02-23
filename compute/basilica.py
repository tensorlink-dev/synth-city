"""
Basilica GPU client — thin wrapper around the ``basilica-sdk`` package that
restricts rentals to a configurable set of cheap GPU offerings, and manages
Docker-based GPU deployments for training.

The official SDK talks to the Basilica secure-cloud marketplace (Hyperstack,
Verda, etc.).  This module adds:

* **Price filtering** — only offerings ≤ ``BASILICA_MAX_HOURLY_RATE`` are shown.
* **GPU-type allowlist** — only GPU types in ``BASILICA_ALLOWED_GPU_TYPES``.
* **Convenience helpers** — ``rent_cheapest()`` picks the cheapest available GPU.
* **Deployments** — Docker-image-based GPU pods via ``create_deployment()``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

from basilica import BasilicaClient as _SdkClient
from basilica import DeploymentResponse, GpuOffering, SecureCloudRentalResponse

from config import (
    BASILICA_ALLOWED_GPU_TYPES,
    BASILICA_API_TOKEN,
    BASILICA_API_URL,
    BASILICA_DEPLOY_GPU_MODELS,
    BASILICA_DEPLOY_IMAGE,
    BASILICA_DEPLOY_MIN_GPU_MEMORY_GB,
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
            if o.hourly_rate is not None
            and o.gpu_type is not None
            and float(o.hourly_rate) <= self.max_hourly_rate
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
            "Started rental %s @ $%s/hr (provider=%s, status=%s)",
            resp.rental_id, resp.hourly_cost, resp.provider, resp.status,
        )
        return resp

    def rent_cheapest(
        self,
        ssh_public_key_id: str | None = None,
    ) -> SecureCloudRentalResponse:
        """Rent the cheapest currently-available GPU within the budget.

        Tries offerings from cheapest to most expensive.  Some provider
        offerings may be incompatible (e.g. "Operating system is not valid
        for this instance type") — those are skipped automatically.

        Raises ``RuntimeError`` if no offerings match or all fail.
        """
        offerings = self.list_cheap_gpus()
        if not offerings:
            raise RuntimeError(
                f"No GPU offerings available within ${self.max_hourly_rate}/hr "
                f"for types {self.allowed_gpu_types}"
            )

        errors: list[tuple[str, str]] = []
        for offering in offerings:
            logger.info(
                "Trying offering: %s %s @ $%s/hr (id=%s)",
                offering.gpu_type,
                "(Spot)" if offering.is_spot else "",
                offering.hourly_rate,
                offering.id,
            )
            try:
                return self.rent_gpu(offering.id, ssh_public_key_id=ssh_public_key_id)
            except Exception as exc:
                msg = str(exc)
                errors.append((offering.id, msg))
                logger.warning(
                    "Offering %s failed: %s — trying next offering",
                    offering.id,
                    msg,
                )

        # All offerings failed
        details = "; ".join(f"{oid}: {err}" for oid, err in errors)
        raise RuntimeError(
            f"All {len(offerings)} GPU offerings failed. Errors: {details}"
        )

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
        """Get detailed status for a single secure-cloud GPU rental (includes SSH info).

        Uses ``list_secure_cloud_rentals()`` to look up the rental, because the
        SDK does not expose a ``get_secure_cloud_rental(id)`` endpoint.
        The ``ssh_command`` field is a plain string like ``"ssh user@host -p port"``.
        """
        resp = self._client.list_secure_cloud_rentals()
        rentals = resp.rentals if hasattr(resp, "rentals") else []
        rental = next((r for r in rentals if r.rental_id == rental_id), None)
        if rental is None:
            raise RuntimeError(
                f"Rental {rental_id!r} not found in list_secure_cloud_rentals(). "
                "It may have already been stopped or the ID is wrong."
            )
        return {
            "rental_id": rental.rental_id,
            "status": getattr(rental, "status", None),
            "ssh_command": getattr(rental, "ssh_command", None),
            "ip_address": getattr(rental, "ip_address", None),
            "hourly_cost": getattr(rental, "hourly_cost", None),
            "created_at": getattr(rental, "created_at", None),
        }

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """Return account balance information."""
        return self._client.get_balance()

    # ------------------------------------------------------------------
    # Docker-based deployments
    # ------------------------------------------------------------------

    # Retry parameters for transient API errors
    _API_MAX_RETRIES = 3
    _API_INITIAL_BACKOFF = 5  # seconds
    _API_BACKOFF_FACTOR = 2

    def create_deployment(
        self,
        name: str,
        image: str | None = None,
        gpu_count: int = 1,
        gpu_models: list[str] | None = None,
        min_gpu_memory_gb: int | None = None,
        env: dict[str, str] | None = None,
        port: int = 8378,
        cpu: str = "2000m",
        memory: str = "8Gi",
        storage: str | None = "10Gi",
    ) -> DeploymentResponse:
        """Create a GPU deployment running a Docker image.

        Uses the Basilica deployments API to spin up a container with GPU
        access.  The container should expose an HTTP server on *port*.

        Retries up to 3 times with exponential backoff on transient API
        errors (HTTP 5xx, connection failures).

        Returns a ``DeploymentResponse`` with ``url``, ``instance_name``,
        ``phase``, etc.
        """
        image = image or BASILICA_DEPLOY_IMAGE
        gpu_models = gpu_models or BASILICA_DEPLOY_GPU_MODELS or None
        if min_gpu_memory_gb is None:
            min_gpu_memory_gb = BASILICA_DEPLOY_MIN_GPU_MEMORY_GB

        last_exc: Exception | None = None
        for attempt in range(self._API_MAX_RETRIES):
            try:
                resp = self._client.create_deployment(
                    instance_name=name,
                    image=image,
                    port=port,
                    gpu_count=gpu_count,
                    gpu_models=gpu_models,
                    min_gpu_memory_gb=min_gpu_memory_gb,
                    env=env or {},
                    cpu=cpu,
                    memory=memory,
                    storage=storage,
                    public=True,
                )
                logger.info(
                    "Created deployment %s (image=%s, phase=%s, url=%s)",
                    resp.instance_name, image, resp.phase, resp.url,
                )
                return resp
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                # Only retry on transient/server errors, not client errors
                is_transient = (
                    "500" in err_str
                    or "502" in err_str
                    or "503" in err_str
                    or "504" in err_str
                    or "connection" in err_str
                    or "timeout" in err_str
                    or "internal server error" in err_str
                )
                if not is_transient:
                    raise

                backoff = self._API_INITIAL_BACKOFF * (
                    self._API_BACKOFF_FACTOR ** attempt
                )
                logger.warning(
                    "create_deployment attempt %d/%d failed (%s) — "
                    "retrying in %ds",
                    attempt + 1,
                    self._API_MAX_RETRIES,
                    exc,
                    backoff,
                )
                if attempt < self._API_MAX_RETRIES - 1:
                    time.sleep(backoff)

        # All retries exhausted
        raise RuntimeError(
            f"create_deployment failed after {self._API_MAX_RETRIES} attempts: "
            f"{last_exc}"
        ) from last_exc

    def get_deployment(self, name: str) -> DeploymentResponse:
        """Get the current status of a deployment."""
        return self._client.get_deployment(name)

    def get_deployment_logs(self, name: str, tail: int | None = 100) -> str:
        """Retrieve logs from a deployment."""
        return self._client.get_deployment_logs(name, tail=tail)

    def delete_deployment(self, name: str) -> dict[str, Any]:
        """Delete a deployment and free its resources."""
        resp = self._client.delete_deployment(name)
        logger.info("Deleted deployment %s", name)
        return {
            "instance_name": getattr(resp, "instance_name", name),
            "status": "deleted",
            "message": getattr(resp, "message", ""),
        }

    def list_deployments(self) -> list[dict[str, Any]]:
        """List all deployments for this account."""
        resp = self._client.list_deployments()
        items = resp.deployments if hasattr(resp, "deployments") else []
        return [
            {
                "instance_name": getattr(d, "instance_name", None),
                "phase": getattr(d, "phase", None),
                "url": getattr(d, "url", None),
                "created_at": getattr(d, "created_at", None),
            }
            for d in items
        ]

    # ------------------------------------------------------------------
    # SSH key management
    # ------------------------------------------------------------------

    def ensure_ssh_key(self, name: str = "synth-city", public_key_path: str | None = None) -> str:
        """Ensure the Basilica-registered SSH key matches the local key.

        If no local key exists, one is generated automatically
        (``ssh-keygen -t ed25519``).  If the Basilica-registered key doesn't
        match the local public key (stale key from a previous machine or
        regenerated keypair), the old key is deleted and the current one is
        registered.

        Returns the SSH key ID that matches the local private key.
        """
        pub_path = os.path.expanduser(public_key_path or "~/.ssh/id_ed25519.pub")
        if not pub_path.endswith(".pub"):
            raise ValueError(
                f"public_key_path must end with '.pub', got {pub_path!r}. "
                "Pass the public key path, not the private key path."
            )
        priv_path = pub_path[:-4]  # strip ".pub"

        # Generate a keypair when neither file exists.
        if not os.path.exists(pub_path):
            if not os.path.exists(priv_path):
                logger.info("No SSH key found at %s — generating ed25519 keypair", priv_path)
                os.makedirs(os.path.dirname(priv_path), exist_ok=True)
                subprocess.run(
                    [
                        "ssh-keygen", "-t", "ed25519",
                        "-f", priv_path,
                        "-N", "",           # empty passphrase
                        "-C", "synth-city",
                    ],
                    check=True,
                    capture_output=True,
                )
            else:
                raise FileNotFoundError(
                    f"Private key exists at {priv_path} but public key is missing at {pub_path}. "
                    "Regenerate the public key with: ssh-keygen -y -f <private_key> > <public_key>"
                )

        with open(pub_path) as f:
            local_pub_key = f.read().strip()

        existing = self._client.get_ssh_key()
        if existing:
            # Compare the key material (ignore trailing comment / whitespace
            # differences by comparing just the type+data portion).
            remote_parts = (existing.public_key or "").strip().split()[:2]
            local_parts = local_pub_key.split()[:2]
            if remote_parts == local_parts:
                logger.info(
                    "SSH key already registered and matches local key: %s (id=%s)",
                    existing.name, existing.id,
                )
                return existing.id

            # Key mismatch — delete the stale remote key and re-register.
            logger.warning(
                "Registered SSH key %r (id=%s) does NOT match local key at %s. "
                "Deleting stale key and re-registering.",
                existing.name, existing.id, pub_path,
            )
            self._client.delete_ssh_key()

        resp = self._client.register_ssh_key(
            name=name,
            public_key=local_pub_key,
        )
        logger.info("Registered SSH key: %s (id=%s)", resp.name, resp.id)
        return resp.id
