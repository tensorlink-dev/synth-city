"""
Live leaderboard dashboard — lightweight HTTP server that serves a
real-time CRPS leaderboard page with auto-refresh.

Reads leaderboard state from the JSON file written by the runner
and serves it as both a human-readable HTML page and a JSON API.

Endpoints:
    GET /             HTML dashboard with auto-refresh
    GET /api/scores   JSON leaderboard data
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from config import WORKSPACE_DIR

logger = logging.getLogger(__name__)

DEFAULT_LEADERBOARD_PATH = WORKSPACE_DIR / "live_validator" / "leaderboard.json"

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synth-City Live Validator</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0d1117; color: #c9d1d9;
    padding: 24px; max-width: 1000px; margin: 0 auto;
  }
  h1 { color: #58a6ff; font-size: 1.5em; margin-bottom: 4px; }
  .subtitle { color: #8b949e; font-size: 0.85em; margin-bottom: 20px; }
  .meta {
    display: flex; gap: 24px; margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .meta-item {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; padding: 12px 16px; min-width: 140px;
  }
  .meta-label { color: #8b949e; font-size: 0.75em; text-transform: uppercase; }
  .meta-value { color: #f0f6fc; font-size: 1.3em; font-weight: bold; }
  .meta-value.leader { color: #3fb950; }
  table {
    width: 100%; border-collapse: collapse; margin-top: 12px;
    background: #161b22; border-radius: 6px; overflow: hidden;
  }
  th {
    text-align: left; padding: 10px 12px; background: #21262d;
    color: #8b949e; font-size: 0.8em; text-transform: uppercase;
    border-bottom: 1px solid #30363d;
  }
  td {
    padding: 10px 12px; border-bottom: 1px solid #21262d;
    font-size: 0.9em;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover { background: #1c2128; }
  .rank { color: #8b949e; font-weight: bold; }
  .rank-1 { color: #f0c000; }
  .rank-2 { color: #aaa; }
  .rank-3 { color: #cd7f32; }
  .model-name { color: #f0f6fc; font-weight: 600; }
  .baseline { color: #8b949e; }
  .research { color: #d2a8ff; }
  .tag {
    font-size: 0.7em; padding: 2px 6px; border-radius: 3px;
    text-transform: uppercase; font-weight: bold;
  }
  .tag-baseline { background: #1f2937; color: #9ca3af; }
  .tag-research { background: #2d1b4e; color: #d2a8ff; }
  .crps { font-variant-numeric: tabular-nums; }
  .crps-best { color: #3fb950; font-weight: bold; }
  .na { color: #484f58; }
  .footer {
    margin-top: 20px; color: #484f58; font-size: 0.75em;
    text-align: center;
  }
  .per-asset { font-size: 0.75em; color: #8b949e; margin-top: 4px; }
</style>
</head>
<body>

<h1>Synth-City Live Validator</h1>
<p class="subtitle">GBM &amp; Heston baselines vs research models &mdash; live CRPS scoring</p>

<div class="meta" id="meta"></div>
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Type</th>
      <th>Avg CRPS</th>
      <th>Best CRPS</th>
      <th>Scores</th>
      <th>Per-Asset</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<p class="footer" id="footer">Loading...</p>

<script>
function fmt(v) {
  if (v === null || v === undefined || v >= 1e6) return '<span class="na">N/A</span>';
  return v.toFixed(4);
}

function renderPerAsset(obj) {
  if (!obj || Object.keys(obj).length === 0) return '';
  return Object.entries(obj)
    .map(([a, v]) => a + ':' + v.toFixed(2))
    .join(' &middot; ');
}

async function refresh() {
  try {
    const resp = await fetch('/api/scores');
    const data = await resp.json();
    const ranking = data.ranking || [];
    const bestCrps = data.best_crps;

    // Meta cards
    document.getElementById('meta').innerHTML = `
      <div class="meta-item">
        <div class="meta-label">Leader</div>
        <div class="meta-value leader">${data.leader || 'N/A'}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Best CRPS</div>
        <div class="meta-value">${fmt(bestCrps)}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Models</div>
        <div class="meta-value">${ranking.length}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Total Entries</div>
        <div class="meta-value">${data.entries_count || 0}</div>
      </div>
    `;

    // Table rows
    const rows = ranking.map(r => {
      const rankCls = r.rank <= 3 ? ' rank-' + r.rank : '';
      const tag = r.is_baseline
        ? '<span class="tag tag-baseline">baseline</span>'
        : '<span class="tag tag-research">research</span>';
      const avgCls = r.rolling_avg_crps === bestCrps
        ? 'crps crps-best' : 'crps';
      return `<tr>
        <td class="rank${rankCls}">#${r.rank}</td>
        <td class="model-name">${r.model}</td>
        <td>${tag}</td>
        <td class="${avgCls}">${fmt(r.rolling_avg_crps)}</td>
        <td class="crps">${fmt(r.best_crps)}</td>
        <td>${r.total_scores}</td>
        <td class="per-asset">${renderPerAsset(r.per_asset)}</td>
      </tr>`;
    }).join('');
    const empty = '<tr><td colspan=7 class="na">No scores yet</td></tr>';
    document.getElementById('tbody').innerHTML = rows || empty;

    const ts = data.last_updated
      ? new Date(data.last_updated).toLocaleString()
      : 'unknown';
    document.getElementById('footer').textContent =
      'Last updated: ' + ts + ' · Auto-refreshes every 10s';
  } catch (e) {
    document.getElementById('footer').textContent = 'Error loading data: ' + e.message;
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the leaderboard dashboard."""

    leaderboard_path: Path = DEFAULT_LEADERBOARD_PATH

    def do_GET(self) -> None:
        if self.path == "/api/scores":
            self._serve_json()
        elif self.path in ("/", "/index.html"):
            self._serve_html()
        else:
            self.send_error(404)

    def _serve_json(self) -> None:
        try:
            data = json.loads(self.leaderboard_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {
                "ranking": [], "entries_count": 0,
                "leader": None, "best_crps": None,
            }

        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        body = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("Dashboard: %s", format % args)


def start_dashboard(
    port: int = 8378,
    leaderboard_path: Path | None = None,
) -> HTTPServer:
    """Start the dashboard HTTP server in a background thread.

    Returns the server instance (call .shutdown() to stop).
    """
    path = leaderboard_path or DEFAULT_LEADERBOARD_PATH
    DashboardHandler.leaderboard_path = path

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(
        target=server.serve_forever, daemon=True,
        name="dashboard",
    )
    thread.start()
    logger.info("Dashboard running at http://localhost:%d", port)
    return server
