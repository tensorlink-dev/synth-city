"""synth-city — thin wrapper for backwards compatibility with ``python main.py``."""

from __future__ import annotations

from cli.app import main

if __name__ == "__main__":
    main()
