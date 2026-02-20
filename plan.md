# Basilica GPU Integration Plan

## Goal

Replace the existing placeholder `BasilicaClient` in `compute/basilica.py` and the stub `submit_basilica_job` tool with a real integration using the official `basilica-sdk` (v0.20.2). Restrict GPU selection to only the cheapest offerings (V100 Spot/On-demand, RTX-A4000, 2x RTX-A4000, RTX-A6000 Spot — all ≤$0.44/hr).

---

## Current State

- **`compute/basilica.py`** — Hand-rolled HTTP client (`BasilicaClient`) using `httpx` against a custom REST API (`/v1/jobs/...`). This does NOT match the real `basilica-sdk` API at all.
- **`pipeline/tools/training_tools.py`** — `submit_basilica_job()` is a placeholder that just returns a JSON spec without actually submitting anything.
- **`config.py`** — Has `BASILICA_API_KEY` and `BASILICA_ENDPOINT` (endpoint defaults to `https://api.basilica.tplr.ai`, but the real SDK uses `https://api.basilica.ai` and the env var `BASILICA_API_TOKEN`).
- **Trainer agent** — Does not include any Basilica tools in its toolset.

## basilica-sdk API Summary (v0.20.2)

The real SDK exposes two GPU rental paths:

1. **Decentralized rentals** (`start_rental`) — rents from Bittensor miner nodes
2. **Secure cloud rentals** (`list_secure_cloud_gpus` / `start_secure_cloud_rental`) — rents from datacenter providers (Hyperstack, Verda, etc.) with `GpuOffering` objects containing `id`, `gpu_type`, `hourly_rate`, `provider`, `is_spot`, `region`, `gpu_count`, `vcpu_count`, `system_memory_gb`, `storage_gb`

The user's GPU table maps to **secure cloud rentals**. The SDK also supports:
- `deploy()` / `create_deployment()` — containerized app deployments
- SSH key management (`register_ssh_key`, `get_ssh_key`, `delete_ssh_key`)
- Balance checking (`get_balance`)

**Auth**: `BASILICA_API_TOKEN` env var or `api_key=` constructor param. API URL defaults to `https://api.basilica.ai`.

---

## Plan

### Step 1: Update `config.py`

- Add `BASILICA_API_TOKEN` (the SDK's expected env var name), keep `BASILICA_API_KEY` as a fallback alias
- Add `BASILICA_API_URL` (SDK expects this, defaults to `https://api.basilica.ai`)
- Add `BASILICA_MAX_HOURLY_RATE` — budget cap, default `0.44` (the max price from the user's table)
- Add `BASILICA_ALLOWED_GPU_TYPES` — allowlist of cheap GPU types

### Step 2: Rewrite `compute/basilica.py`

Replace the hand-rolled HTTP client with a thin wrapper around the real `basilica-sdk`. The new module will:

- Import `basilica.BasilicaClient` (from the SDK) and wrap it
- Provide a `BasilicaGPUClient` class with these methods:
  - `list_cheap_gpus()` — calls `client.list_secure_cloud_gpus()`, filters to offerings ≤ max hourly rate and matching the allowed GPU type list
  - `rent_gpu(offering_id)` — calls `client.start_secure_cloud_rental(offering_id)`
  - `stop_gpu(rental_id)` — calls `client.stop_secure_cloud_rental(rental_id)`
  - `list_active_rentals()` — calls `client.list_secure_cloud_rentals()`
  - `get_rental_status(rental_id)` — calls `client.get_rental(rental_id)`
  - `get_balance()` — calls `client.get_balance()`
  - `ensure_ssh_key()` — registers SSH key if not already registered
  - `rent_cheapest_gpu()` — convenience: lists cheap GPUs, picks the cheapest available, rents it
- Define the allowed GPU filter config:
  ```python
  ALLOWED_OFFERINGS = [
      ("TESLA V100", True),    # V100 Spot @ $0.05/hr
      ("TESLA V100", False),   # V100 On-demand @ $0.15/hr
      ("RTX-A4000", False),    # 1x @ $0.16/hr
      ("RTX-A4000", False),    # 2x @ $0.33/hr (gpu_count=2)
      ("RTX-A6000", True),     # A6000 Spot @ $0.44/hr
  ]
  ```
- Filtering logic: match on `gpu_type` substring + `is_spot` flag + `hourly_rate <= max_rate`, sorted by `hourly_rate` ascending

### Step 3: Rewrite `pipeline/tools/training_tools.py` Basilica tools

Replace the stub `submit_basilica_job` with real tools:

- **`list_available_gpus()`** — Lists cheap GPU offerings with pricing. Calls `BasilicaGPUClient.list_cheap_gpus()`.
- **`rent_gpu(offering_id: str)`** — Rents a specific GPU offering. Returns rental ID, SSH command, IP, hourly cost.
- **`rent_cheapest_gpu()`** — One-click: rent the cheapest available GPU. Returns rental details.
- **`list_active_rentals()`** — Shows all current GPU rentals with status and cost.
- **`stop_gpu_rental(rental_id: str)`** — Stops a rental and returns total cost.
- **`check_gpu_balance()`** — Returns current account balance.

Remove the old `submit_basilica_job` tool.

### Step 4: Wire tools into the Trainer agent

Update `pipeline/agents/trainer.py` to include the new Basilica tools in its `build_tools()` method:

```python
tool_names = [
    # ... existing tools ...
    # GPU rental
    "list_available_gpus",
    "rent_gpu",
    "rent_cheapest_gpu",
    "list_active_rentals",
    "stop_gpu_rental",
    "check_gpu_balance",
]
```

Add `import pipeline.tools.training_tools  # noqa: F401` if not already triggered.

### Step 5: Update `.env.example` and `config.py` docs

- Replace `BASILICA_API_KEY` / `BASILICA_ENDPOINT` with `BASILICA_API_TOKEN` / `BASILICA_API_URL`
- Add `BASILICA_MAX_HOURLY_RATE=0.44`
- Document the cheap GPU strategy

### Step 6: Update `docs/SETUP_GUIDE.md`

Update the Basilica section to reflect:
- `pip install basilica-sdk` as a dependency
- New env vars (`BASILICA_API_TOKEN`)
- New usage examples with the real SDK
- The cheap GPU filtering strategy

---

## Files Modified

| File | Action |
|------|--------|
| `config.py` | Update env vars for basilica-sdk |
| `compute/basilica.py` | Full rewrite — wrap `basilica-sdk` |
| `pipeline/tools/training_tools.py` | Replace stub with real GPU rental tools |
| `pipeline/agents/trainer.py` | Add GPU tools to trainer toolset |
| `.env.example` | Update Basilica env var names |
| `docs/SETUP_GUIDE.md` | Update Basilica docs |
| `pyproject.toml` | Add `basilica-sdk` dependency |

## Constraints

- Only allow GPUs ≤ $0.44/hr (the 5 offerings from the user's table)
- Filter by GPU type name + spot flag to match exactly those offerings
- Sort by price ascending so the cheapest is always preferred
- Keep `run_training_local` and `run_python` tools unchanged
