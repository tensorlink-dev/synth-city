"""
Publish the synth-city OpenClaw skill to ClawHub with retry on rate limit.

Usage::

    python integrations/openclaw/publish.py
    python integrations/openclaw/publish.py --version 0.1.2
    python integrations/openclaw/publish.py --skill-dir ./integrations/openclaw/skill --version 0.1.2

Wraps ``clawhub publish`` and automatically retries with exponential backoff
when the ClawHub registry returns a rate limit error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SKILL_DIR = Path(__file__).parent / "skill"

# Retry settings
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 2.0  # seconds; doubles on each retry

# Strings that indicate a transient rate-limit response from clawhub
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "too many requests")


def _is_rate_limit_error(output: str) -> bool:
    lower = output.lower()
    return any(marker in lower for marker in _RATE_LIMIT_MARKERS)


def publish(skill_dir: Path, version: str | None) -> int:
    """Run ``clawhub publish`` with exponential-backoff retry on rate limits.

    Returns the final process exit code.
    """
    cmd: list[str] = ["clawhub", "publish", str(skill_dir)]
    if version:
        cmd += ["--version", version]

    backoff = _INITIAL_BACKOFF

    for attempt in range(1, _MAX_RETRIES + 1):
        print(f"Publishing (attempt {attempt}/{_MAX_RETRIES}): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)  # noqa: S603

        if result.returncode == 0:
            return 0

        # Collect any output that may have been captured
        combined = (result.stdout or "") + (result.stderr or "")

        if not _is_rate_limit_error(combined) or attempt == _MAX_RETRIES:
            # Non-rate-limit failure, or we've exhausted all retries
            if _is_rate_limit_error(combined):
                print(
                    f"\nRate limit still active after {_MAX_RETRIES} attempts. "
                    "Try again later.",
                    file=sys.stderr,
                )
            return result.returncode

        print(f"Rate limit exceeded — retrying in {backoff:.0f}s …", file=sys.stderr)
        time.sleep(backoff)
        backoff *= 2

    return 1  # unreachable, but satisfies type checkers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish synth-city skill to ClawHub (retries on rate limit)"
    )
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=SKILL_DIR,
        help=f"Path to skill directory (default: {SKILL_DIR})",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version to publish (overrides clawhub.json; e.g. 0.1.2)",
    )
    args = parser.parse_args()

    if not args.skill_dir.is_dir():
        print(f"ERROR: skill directory not found: {args.skill_dir}", file=sys.stderr)
        sys.exit(1)

    sys.exit(publish(args.skill_dir, args.version))


if __name__ == "__main__":
    main()
