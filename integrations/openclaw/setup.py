"""
Install the synth-city skill into an OpenClaw workspace.

Usage::

    python integrations/openclaw/setup.py
    python integrations/openclaw/setup.py --workspace ~/.openclaw/workspace

This copies the SKILL.md and tools into the OpenClaw workspace skills directory
so the OpenClaw agent automatically discovers synth-city capabilities.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"
SKILL_NAME = "synth-city"

# Files that must exist in the skill source directory
REQUIRED_FILES = ["SKILL.md", "tools.py"]

# Optional files copied when present (e.g. ClawHub publishing metadata)
OPTIONAL_FILES = ["clawhub.json"]


def install_skill(workspace: Path) -> bool:
    """Copy skill files into the OpenClaw workspace.

    Returns ``True`` on success, ``False`` on failure.
    """
    skill_dir = workspace / "skills" / SKILL_NAME
    src_dir = Path(__file__).parent / "skill"

    # Validate all required source files exist before creating anything
    missing = [f for f in REQUIRED_FILES if not (src_dir / f).exists()]
    if missing:
        print(
            f"ERROR: Missing source files in {src_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )
        return False

    # Create target directory
    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: Cannot create skill directory {skill_dir}: {exc}", file=sys.stderr)
        return False

    # Copy each file and verify
    for filename in REQUIRED_FILES:
        src = src_dir / filename
        dst = skill_dir / filename
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            print(f"ERROR: Failed to copy {src} → {dst}: {exc}", file=sys.stderr)
            return False

        if not dst.exists():
            print(f"ERROR: Copy verification failed — {dst} does not exist", file=sys.stderr)
            return False

        print(f"  Installed {filename} → {dst}")

    # Copy optional files (e.g. clawhub.json) when present
    for filename in OPTIONAL_FILES:
        src = src_dir / filename
        if not src.exists():
            continue
        dst = skill_dir / filename
        try:
            shutil.copy2(src, dst)
            print(f"  Installed {filename} → {dst}")
        except OSError as exc:
            print(f"  Warning: Could not copy optional file {filename}: {exc}", file=sys.stderr)

    print(f"\nsynth-city skill installed to: {skill_dir}")
    print("\nNext steps:")
    print("  1. Start the bridge:  python main.py bridge")
    print("  2. Start OpenClaw:    openclaw start")
    print("  3. Chat:              'Hey, list the available blocks for SN50'")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Install synth-city skill into OpenClaw workspace")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"OpenClaw workspace path (default: {DEFAULT_WORKSPACE})",
    )
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"OpenClaw workspace not found at {args.workspace}")
        print("Make sure OpenClaw is installed: npm install -g openclaw@latest && openclaw onboard")
        sys.exit(1)

    if not install_skill(args.workspace):
        sys.exit(1)


if __name__ == "__main__":
    main()
