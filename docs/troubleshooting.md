# Troubleshooting

Common errors, diagnostic steps, and fixes for synth-city. Organized by category for quick lookup.

---

## Installation Issues

### ModuleNotFoundError: No module named 'research'

**Cause:** open-synth-miner is not installed. synth-city's research tools depend on it.

**Fix:**

```bash
# Option A: editable install from local clone
git clone https://github.com/tensorlink-dev/open-synth-miner.git ../open-synth-miner
pip install -e ../open-synth-miner

# Option B: install from GitHub
pip install "open-synth-miner @ git+https://github.com/tensorlink-dev/open-synth-miner.git"
```

**Verify:**

```bash
python -c "from research import ResearchSession; print('OK')"
```

### ImportError: cannot import name 'ResearchSession'

**Cause:** open-synth-miner is installed but at an incompatible version. The import path may have changed.

**Fix:** Update to the latest version:

```bash
cd ../open-synth-miner
git pull
pip install -e .
```

synth-city tries three import paths: `osa.research.agent_api`, `src.research.agent_api`, and `research.agent_api`. If none work, the version is too old.

### torch not found or CUDA unavailable

**Cause:** PyTorch was installed without CUDA support, or CUDA drivers are missing.

**Diagnose:**

```bash
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"
nvidia-smi
```

**Fix:** Install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU-only training works but is significantly slower. For production, use Basilica GPU deployments instead.

---

## LLM and API Errors

### CHUTES_API_KEY not set

**Cause:** The `.env` file is missing or `CHUTES_API_KEY` is not defined.

**Fix:**

```bash
cp .env.example .env
# Edit .env and set CHUTES_API_KEY=your-key
```

### LLM API connection errors / timeouts

**Diagnose:**

```bash
curl https://llm.chutes.ai/v1/models -H "Authorization: Bearer $CHUTES_API_KEY"
```

**Common causes:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Chutes endpoint down | Wait and retry; check [chutes.ai](https://chutes.ai/) status |
| 401 Unauthorized | Invalid API key | Verify `CHUTES_API_KEY` in `.env` |
| 429 Too Many Requests | Rate limited | Reduce pipeline concurrency or wait |
| Timeout (300s) | Model overloaded | Try a different model or retry later |
| Model not found | Model ID changed | Check available models at the endpoint |

The client retries transient errors automatically (up to 7 total attempts with exponential backoff: 2s, 4s, 8s, 16s). If all retries fail, a `RuntimeError` is raised.

### Using a different LLM provider

Change `CHUTES_BASE_URL` to any OpenAI-compatible endpoint:

```bash
# OpenRouter
CHUTES_BASE_URL=https://openrouter.ai/api/v1
CHUTES_API_KEY=your-openrouter-key

# Local vLLM
CHUTES_BASE_URL=http://localhost:8000/v1
CHUTES_API_KEY=dummy

# Ollama
CHUTES_BASE_URL=http://localhost:11434/v1
CHUTES_API_KEY=ollama
```

Ensure the models you configure (`DEFAULT_MODEL`, `PLANNER_MODEL`, etc.) are available on your provider.

---

## Pipeline Issues

### Agent gets stuck in a loop

**Symptoms:** The Debugger produces the same config repeatedly. Logs show identical tool calls.

**Built-in protection:** The orchestrator compares experiment configs between attempts. If unchanged, it injects a critical warning forcing the agent to try a different approach.

**If the problem persists:**

1. **Increase temperature** — start higher to encourage creativity:
   ```bash
   synth-city pipeline --temperature 0.3
   ```

2. **Increase retries** — give the agent more chances:
   ```bash
   synth-city pipeline --retries 10
   ```

3. **Establish baselines first** — run a sweep so the Planner has history to reference:
   ```bash
   synth-city sweep --epochs 2
   ```

4. **Check agent logs** — look for repeated tool calls with identical arguments in `logs/`.

### Pipeline fails at Planner stage

**Common causes:**

- **No components found:** open-synth-miner not installed or registry empty. Verify with `synth-city client blocks`.
- **LLM can't follow instructions:** Try a more capable model (`PLANNER_MODEL=Qwen/Qwen3-235B-A22B`).
- **History load failure:** Hippius unreachable. Not fatal — the Planner works without history, just less informed.

### Pipeline fails at Trainer stage

**Common causes:**

- **GPU deployment fails:** Check Basilica balance with `check_gpu_balance()`. Verify `BASILICA_API_TOKEN`.
- **Training OOM:** Reduce batch size or d_model. See [Out of Memory](#out-of-memory-oom).
- **Experiment config invalid:** The Trainer should validate before running. If it doesn't, check the Planner's output.

### Non-recoverable errors stop retries

The orchestrator recognizes certain errors as non-recoverable and stops retrying:

- `ModuleNotFoundError` — missing dependency
- `ImportError` — broken installation
- Other environment errors

These require fixing the environment, not retrying the agent. Check the error message and install missing dependencies.

---

## CRPS and Model Quality

### CRPS scores are poor (high values)

CRPS is lower-is-better. High scores indicate poor forecast calibration.

**Tuning checklist:**

| Parameter | Research Default | Better Quality |
|-----------|-----------------|----------------|
| `RESEARCH_N_PATHS` | 100 | 1000 |
| `RESEARCH_EPOCHS` | 1 | 5–10 |
| `RESEARCH_D_MODEL` | 32 | 64–128 |
| `RESEARCH_SEQ_LEN` | 32 | 64–128 |
| `RESEARCH_BATCH_SIZE` | 4 | 16–64 |

**Architecture advice:**

- Start with known-good presets: `synth-city sweep --epochs 3`
- More expressive heads (SDEHead, FlowHead) generally outperform simple ones (GBMHead)
- Deeper stacks (more blocks) can help but increase training time
- Check that the prediction horizon matches SN50 requirements (288 steps for 24h)

### NaN or Inf in predictions

**Cause:** Training divergence, usually from a learning rate that's too high or an unstable architecture.

**Fix:**

- Lower the learning rate: `RESEARCH_LR=0.0001`
- Use gradient clipping (configured in the model head)
- Try a different head — GBMHead is the most stable baseline
- Check for degenerate data (all-zero features, missing values)

### check_shapes validation fails

**Cause:** The model's `generate_paths()` output doesn't match SN50 requirements.

**Required output shape:** `[1000, 289]` for 24h horizon

**Common shape mismatches:**

| Got | Expected | Fix |
|-----|----------|-----|
| `[100, 289]` | `[1000, 289]` | Set `n_paths=1000` for production |
| `[1000, 288]` | `[1000, 289]` | Include t0 in the path (289 = 288 + 1) |
| `[1000, 12]` | `[1000, 289]` | Set `horizon=288` for production |

---

## Bridge Server Issues

### Bridge server won't start

**Diagnose:**

```bash
# Check if port is already in use
lsof -i :8377
# or
ss -tlnp | grep 8377
```

**Fix:**

- Kill the existing process, or use a different port: `synth-city bridge --port 9000`
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Bridge returns 403 Forbidden

**Cause:** API key mismatch.

**Fix:** Ensure the `X-API-Key` header matches `BRIDGE_API_KEY` in the server's `.env`:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8377/health
```

### Bridge connection timeout from remote client

**Cause:** Bridge is bound to `127.0.0.1` (localhost only).

**Fix:** Either:

1. Bind to all interfaces: `BRIDGE_HOST=0.0.0.0` (requires `BRIDGE_API_KEY`)
2. Use SSH tunnel: `ssh -L 8377:localhost:8377 user@server -N`

### Bot session isolation not working

**Cause:** Missing `X-Bot-Id` header. All requests without it use the "default" session.

**Fix:** Always include the header:

```bash
curl -H "X-Bot-Id: my-bot" http://localhost:8377/components/blocks
```

---

## GPU and Training Issues

### Out of Memory (OOM)

**Symptoms:** `CUDA out of memory`, `RuntimeError: CUDA error`, or process killed.

**Immediate fixes:**

```bash
RESEARCH_BATCH_SIZE=2      # reduce from 4
RESEARCH_D_MODEL=16        # reduce from 32
RESEARCH_N_PATHS=50        # reduce from 100
```

**Better solution:** Use Basilica GPU deployments for larger experiments. The Trainer agent does this automatically.

### Basilica deployment stuck in Pending

**Diagnose:**

```bash
synth-city client deployments
```

**Common causes:**

- **No GPUs available:** All matching GPUs are rented. Wait or adjust `BASILICA_ALLOWED_GPU_TYPES`.
- **Insufficient balance:** Check with `check_gpu_balance()`.
- **Image pull failure:** Verify the Docker image exists in the registry.

**Check logs:**

```bash
# Via bridge client
curl http://localhost:8377/deployments/logs?name=your-deployment
```

### Basilica deployment fails immediately

**Diagnose:** Check deployment logs for startup errors.

**Common causes:**

- **Missing CUDA in image:** Ensure the Docker image has CUDA runtime
- **Import errors:** Missing Python dependencies in the image
- **Port conflict:** Training server can't bind to expected port

### Always clean up deployments

GPU pods cost money per hour. Always delete them when done:

```bash
curl -X DELETE http://localhost:8377/deployments/your-deployment-name
```

The Trainer agent should do this automatically, but verify with `list_deployments` after runs complete.

---

## Storage Issues

### Hippius storage not working (silently skipped)

**Expected behavior:** If Hippius credentials are not set, the pipeline silently skips remote storage. Results remain local in `WORKSPACE_DIR`.

**To enable Hippius:**

```bash
HIPPIUS_ENDPOINT=https://s3.hippius.network
HIPPIUS_ACCESS_KEY=your-access-key
HIPPIUS_SECRET_KEY=your-secret-key
HIPPIUS_BUCKET=synth-city
```

**Diagnose connectivity:**

```bash
python -c "
from pipeline.tools.hippius_store import _get_client
client = _get_client()
print('Connected' if client else 'No client')
"
```

### Hippius endpoint marked unreachable

After 3 consecutive connection failures, the Hippius client marks the endpoint as unreachable and stops trying for the rest of the session. This prevents slow timeouts on every operation.

**Fix:** Restart the process to reset the reachability flag:

```bash
# Restart the bridge server
docker compose restart bridge
```

### S3 bucket doesn't exist

The Hippius client attempts to create the bucket automatically via `_ensure_bucket()`. If this fails, check that your credentials have bucket creation permissions, or create it manually.

---

## Publishing Issues

### HuggingFace publish fails

**Diagnose:**

```bash
huggingface-cli whoami
```

**Common causes:**

| Error | Fix |
|-------|-----|
| Not logged in | Run `huggingface-cli login` |
| No write access to repo | Check `HF_REPO_ID` and repo permissions |
| Repo doesn't exist | Create the repo first on huggingface.co |
| `HF_TOKEN` not set | Set in `.env` or use `huggingface-cli login` |

### Weights & Biases logging fails

```bash
wandb login
wandb status
```

Ensure `WANDB_PROJECT` is set in `.env`.

### Trackio logging fails

Trackio is a local tracking library — it doesn't require authentication. If it fails:

```bash
pip install trackio
```

---

## Docker Issues

### GPU not available in container

**Diagnose:**

```bash
docker run --gpus all nvidia/smi
```

**Requirements:**

- NVIDIA Container Toolkit installed on the host
- `--gpus all` flag passed to Docker
- `docker-compose.yml` includes the `deploy.resources.reservations.devices` section

### Docker image build fails at PyTorch

**Cause:** PyTorch CUDA version mismatch with the base image.

The Dockerfile uses `nvidia/cuda:12.2.0-runtime-ubuntu22.04` and installs PyTorch with CUDA 12.1 index. These are forward-compatible. If your host has older CUDA drivers, you may need to adjust the base image.

---

## Logging and Debugging

### Finding Logs

```
logs/                        # rotating log files (20 MB, 5 backups)
workspace/                   # experiment artifacts
```

Log level is controlled by `LOG_LEVEL` (default: `INFO`). Set to `DEBUG` for maximum verbosity:

```bash
LOG_LEVEL=DEBUG synth-city pipeline
```

### Inspecting Agent Conversations

The full message history is captured in `AgentResult.messages`. When running a single agent:

```bash
synth-city agent --name planner 2>&1 | tee planner-debug.log
```

### Monitoring Pipeline Events

The `pipeline/monitor.py` module emits events for observability:

- `pipeline_start`, `pipeline_end`
- `stage_start`, `stage_end`
- `agent_start`, `agent_end`
- `tool_call`, `tool_result`
- `retry_attempt`

### Bridge Dashboard

Access the HTML dashboard at `http://localhost:8377/dashboard` for real-time monitoring of bot sessions, pipeline status, and experiment results.

---

## Quick Diagnostic Commands

```bash
# Check installation
python -c "from pipeline.orchestrator import PipelineOrchestrator; print('synth-city OK')"
python -c "from research import ResearchSession; print('open-synth-miner OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check configuration
python -c "from config import CHUTES_API_KEY; print('API key:', 'set' if CHUTES_API_KEY else 'MISSING')"

# Check bridge health
curl http://localhost:8377/health

# Check available components
curl http://localhost:8377/components/blocks

# Check Basilica balance
python -c "from compute.basilica import BasilicaGPUClient; print(BasilicaGPUClient().get_balance())"

# Check Hippius connectivity
python -c "from pipeline.tools.hippius_store import load_hippius_history; print(load_hippius_history(1))"

# Lint and test
ruff check .
pytest --tb=short -q
```

---

## Getting Help

If you encounter an issue not covered here:

1. Check the logs in `logs/` for detailed error messages
2. Run the failing operation with `LOG_LEVEL=DEBUG` for maximum verbosity
3. Check the [GitHub issues](https://github.com/tensorlink-dev/synth-city/issues) for known problems
4. Open a new issue with:
   - The exact error message
   - Relevant log output
   - Your Python version and OS
   - Whether open-synth-miner is installed and which version
