"""
Basilica compute client — submit and monitor GPU training jobs on the
Basilica decentralised compute network (Bittensor SN39).

Basilica provides an API for submitting containerised training jobs that
run on miner-provided GPUs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from config import BASILICA_API_KEY, BASILICA_ENDPOINT

logger = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """Specification for a Basilica training job."""

    script_path: str
    gpu_type: str = "A100"
    num_gpus: int = 1
    docker_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    requirements: list[str] | None = None
    env_vars: dict[str, str] | None = None
    timeout_minutes: int = 60
    upload_files: list[str] | None = None


@dataclass
class JobStatus:
    """Status of a submitted Basilica job."""

    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float = 0.0
    output: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0


class BasilicaClient:
    """Client for the Basilica decentralised compute API.

    Usage::

        client = BasilicaClient()
        job_id = client.submit(JobSpec(script_path="train.py"))
        status = client.poll(job_id, timeout=3600)
        if status.status == "completed":
            client.download_artifacts(job_id, "output/")
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.api_key = api_key or BASILICA_API_KEY
        self.endpoint = (endpoint or BASILICA_ENDPOINT).rstrip("/")
        self.http = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )

    def submit(self, spec: JobSpec) -> str:
        """Submit a training job. Returns the job ID."""
        if not self.api_key:
            raise RuntimeError(
                "BASILICA_API_KEY not set. Configure it in .env to submit jobs."
            )

        # Read the training script
        script_content = Path(spec.script_path).read_text(encoding="utf-8")

        payload = {
            "script": script_content,
            "gpu_type": spec.gpu_type,
            "num_gpus": spec.num_gpus,
            "docker_image": spec.docker_image,
            "requirements": spec.requirements or [],
            "env_vars": spec.env_vars or {},
            "timeout_minutes": spec.timeout_minutes,
        }

        resp = self.http.post("/v1/jobs", json=payload)
        resp.raise_for_status()
        data = resp.json()
        job_id = data["job_id"]
        logger.info("Submitted Basilica job: %s", job_id)
        return job_id

    def status(self, job_id: str) -> JobStatus:
        """Check the status of a job."""
        resp = self.http.get(f"/v1/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        return JobStatus(
            job_id=job_id,
            status=data.get("status", "unknown"),
            progress=data.get("progress", 0.0),
            output=data.get("output", ""),
            error=data.get("error", ""),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )

    def poll(
        self,
        job_id: str,
        timeout: int = 3600,
        interval: int = 30,
    ) -> JobStatus:
        """Poll a job until completion or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.status(job_id)
            logger.info(
                "Job %s: %s (%.0f%%, %.0fs elapsed)",
                job_id, status.status, status.progress * 100, status.elapsed_seconds,
            )
            if status.status in ("completed", "failed"):
                return status
            time.sleep(interval)

        logger.warning("Job %s timed out after %ds", job_id, timeout)
        return JobStatus(
            job_id=job_id,
            status="timeout",
            elapsed_seconds=time.time() - start,
        )

    def download_artifacts(self, job_id: str, output_dir: str) -> list[str]:
        """Download output artifacts from a completed job."""
        resp = self.http.get(f"/v1/jobs/{job_id}/artifacts")
        resp.raise_for_status()
        data = resp.json()

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        downloaded = []

        for artifact in data.get("artifacts", []):
            file_resp = self.http.get(f"/v1/jobs/{job_id}/artifacts/{artifact['name']}")
            file_resp.raise_for_status()
            target = out_path / artifact["name"]
            target.write_bytes(file_resp.content)
            downloaded.append(str(target))
            logger.info("Downloaded: %s", target)

        return downloaded

    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        self.http.post(f"/v1/jobs/{job_id}/cancel")
        logger.info("Cancelled job: %s", job_id)
