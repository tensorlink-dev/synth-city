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


def install_skill(workspace: Path) -> None:
    """Copy skill files into the OpenClaw workspace."""
    skill_dir = workspace / "skills" / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Source files
    src_dir = Path(__file__).parent / "skill"
    skill_md = src_dir / "SKILL.md"
    tools_py = src_dir / "tools.py"

    if not skill_md.exists():
        print(f"ERROR: {skill_md} not found", file=sys.stderr)
        sys.exit(1)

    # Copy SKILL.md
    shutil.copy2(skill_md, skill_dir / "SKILL.md")
    print(f"  Installed SKILL.md → {skill_dir / 'SKILL.md'}")

    # Copy tools.py
    if tools_py.exists():
        shutil.copy2(tools_py, skill_dir / "tools.py")
        print(f"  Installed tools.py → {skill_dir / 'tools.py'}")

    print(f"\nsynth-city skill installed to: {skill_dir}")
    print("\nNext steps:")
    print("  1. Start the bridge:  python main.py bridge")
    print("  2. Start OpenClaw:    openclaw start")
    print("  3. Chat:              'Hey, list the available blocks for SN50'")


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

    install_skill(args.workspace)


if __name__ == "__main__":
    main()
