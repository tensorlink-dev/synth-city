"""Startup wrapper for the synth-city GPU training server.

Logs early diagnostics to stdout before launching the server.  If the
server crashes on import, these lines are the only output visible in
``get_deployment_logs()`` and help distinguish "image broken" from
"container never started".
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    build = os.environ.get("BUILD_VERSION", "?")
    print(f"[startup] synth-city-gpu build={build}", flush=True)
    print(f"[startup] python={sys.version}", flush=True)

    # --- torch / CUDA ---
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        print(f"[startup] torch={torch.__version__} cuda={cuda_ok}", flush=True)
        if cuda_ok:
            dev = torch.cuda.get_device_name(0)
            print(f"[startup] gpu={dev}", flush=True)
    except Exception as e:
        print(f"[startup] torch import FAILED: {e}", flush=True)

    # --- ResearchSession ---
    try:
        from osa.research.agent_api import ResearchSession  # noqa: F401

        print("[startup] ResearchSession OK", flush=True)
    except Exception as e:
        print(f"[startup] ResearchSession import FAILED: {e}", flush=True)

    # --- Launch server ---
    print("[startup] launching training server...", flush=True)
    import training_server  # noqa: E402
    import uvicorn  # noqa: E402

    port = int(os.environ.get("TRAINING_SERVER_PORT", "8378"))
    uvicorn.run(training_server.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
