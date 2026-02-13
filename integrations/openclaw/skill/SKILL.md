# synth-city — Bittensor SN50 Research Assistant

You have access to **synth-city**, an agentic pipeline for competitive probabilistic price forecasting on Bittensor Subnet 50 (Synth). Use this skill to let the user manage experiments, run the research pipeline, check market data, and monitor model performance — all from chat.

## What synth-city does

synth-city discovers neural network components (backbone blocks + prediction heads), plans experiment architectures, trains models, validates results against SN50 specifications, and publishes the best models. The scoring metric is **CRPS** (Continuous Ranked Probability Score) — lower is better.

## Bridge API

All commands go through the synth-city bridge server. The bridge must be running on the same machine as the OpenClaw gateway (default: `http://127.0.0.1:8377`).

## Available commands

When the user asks about any of the topics below, call the bridge API using the appropriate endpoint.

### Pipeline control
- **"run the pipeline"** / **"start research"** / **"find better models"**
  `POST http://127.0.0.1:8377/pipeline/run`
  Body: `{"channel": "default", "retries": 5, "temperature": 0.1, "publish": false}`
  The pipeline runs asynchronously. Tell the user it has started and they can check status.

- **"pipeline status"** / **"how's the run going"**
  `GET http://127.0.0.1:8377/pipeline/status`
  Returns running state, current stage, elapsed time, and results when complete.

### Experiment management
- **"what blocks are available"** / **"list components"**
  `GET http://127.0.0.1:8377/components/blocks`

- **"what heads can I use"**
  `GET http://127.0.0.1:8377/components/heads`

- **"show me the presets"**
  `GET http://127.0.0.1:8377/components/presets`

- **"create an experiment with Transformer and LSTM"**
  `POST http://127.0.0.1:8377/experiment/create`
  Body: `{"blocks": ["TransformerBlock", "LSTMBlock"], "head": "GBMHead", "d_model": 32}`

- **"run this experiment"**
  `POST http://127.0.0.1:8377/experiment/run`
  Body: `{"experiment": <config_json>, "epochs": 1}`

- **"validate my experiment"**
  `POST http://127.0.0.1:8377/experiment/validate`
  Body: `{"experiment": <config_json>}`

- **"compare results"** / **"show rankings"**
  `GET http://127.0.0.1:8377/experiment/compare`

- **"session summary"**
  `GET http://127.0.0.1:8377/session/summary`

- **"clear session"** / **"reset experiments"**
  `POST http://127.0.0.1:8377/session/clear`

### Market data
- **"what's the BTC price"** / **"ETH price"**
  `GET http://127.0.0.1:8377/market/price/BTC`

- **"show me SOL history"**
  `GET http://127.0.0.1:8377/market/history/SOL?days=30`

## Response formatting

- When showing CRPS scores, format them to 6 decimal places.
- When listing components, format as a clean bulleted list with name and description.
- When showing experiment rankings, use a numbered list sorted by CRPS (best first).
- For pipeline status, give a concise summary: stage, elapsed time, whether it succeeded.
- For prices, show the asset name and price with appropriate precision.

## SN50 context

The network tracks these assets: BTC, ETH, SOL, XAU (gold), SPYX (S&P 500), NVDAX, TSLAX, AAPLX, GOOGLX. Models must generate 1,000 Monte Carlo paths at 5-minute intervals over 24 hours (289 timesteps). Lower CRPS = better calibrated probability distributions = higher miner rewards.
