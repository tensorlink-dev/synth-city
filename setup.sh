#!/usr/bin/env bash
# synth-city setup — installs synth-city + open-synth-miner in one step.
#
# Usage:
#   ./setup.sh                           # interactive defaults
#   ./setup.sh --osm-path ../open-synth-miner  # existing checkout
#   ./setup.sh --no-venv                 # skip virtualenv (e.g. Docker)
#   ./setup.sh --with-openclaw           # also install OpenClaw skill
#
# What it does:
#   1. Creates a Python virtualenv (unless --no-venv)
#   2. Clones open-synth-miner if not found locally
#   3. Installs open-synth-miner + synth-city in editable mode
#   4. Copies .env.example → .env if .env doesn't exist
#   5. Optionally installs the OpenClaw skill
#
# Requirements: Python 3.10+, git, pip

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSM_PATH=""
VENV_DIR="$SCRIPT_DIR/.venv"
SKIP_VENV=false
WITH_DEV=false
WITH_OPENCLAW=false

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --osm-path)   OSM_PATH="$2"; shift 2 ;;
        --venv-dir)   VENV_DIR="$2"; shift 2 ;;
        --no-venv)    SKIP_VENV=true; shift ;;
        --dev)        WITH_DEV=true; shift ;;
        --with-openclaw) WITH_OPENCLAW=true; shift ;;
        -h|--help)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --osm-path PATH   Path to existing open-synth-miner checkout"
            echo "  --venv-dir PATH   Virtualenv location (default: .venv)"
            echo "  --no-venv         Skip virtualenv creation (use current Python)"
            echo "  --dev             Also install dev dependencies (pytest, ruff, mypy)"
            echo "  --with-openclaw   Install the OpenClaw skill after setup"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---- helpers ----------------------------------------------------------------
info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==>\033[0m $*"; }
fail()  { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ---- pre-flight checks -----------------------------------------------------
info "Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python 3.10+."
command -v git     >/dev/null 2>&1 || fail "git not found."

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PYTHON_MAJOR" -lt 3 ]] || { [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 10 ]]; }; then
    fail "Python 3.10+ required, found $PYTHON_VERSION"
fi
ok "Python $PYTHON_VERSION"

# ---- virtualenv -------------------------------------------------------------
if [[ "$SKIP_VENV" == false ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        info "Creating virtualenv at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        info "Virtualenv already exists at $VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ok "Activated virtualenv: $(which python)"
else
    info "Skipping virtualenv (--no-venv)"
fi

# ---- upgrade pip ------------------------------------------------------------
info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ---- open-synth-miner ------------------------------------------------------
if [[ -z "$OSM_PATH" ]]; then
    # Try sibling directory first
    if [[ -d "$SCRIPT_DIR/../open-synth-miner" ]]; then
        OSM_PATH="$(cd "$SCRIPT_DIR/../open-synth-miner" && pwd)"
        info "Found open-synth-miner at $OSM_PATH"
    else
        OSM_PATH="$SCRIPT_DIR/../open-synth-miner"
        info "Cloning open-synth-miner..."
        git clone https://github.com/tensorlink-dev/open-synth-miner.git "$OSM_PATH"
        OSM_PATH="$(cd "$OSM_PATH" && pwd)"
    fi
fi

if [[ ! -d "$OSM_PATH" ]]; then
    fail "open-synth-miner not found at $OSM_PATH"
fi

ok "open-synth-miner: $OSM_PATH"

# ---- install open-synth-miner ----------------------------------------------
info "Installing open-synth-miner (editable)..."
if [[ -f "$OSM_PATH/pyproject.toml" ]] || [[ -f "$OSM_PATH/setup.py" ]]; then
    pip install -e "$OSM_PATH" --quiet
else
    # Fallback: add to path if no setup file
    warn "No pyproject.toml or setup.py in open-synth-miner — adding to PYTHONPATH"
    export PYTHONPATH="${OSM_PATH}:${PYTHONPATH:-}"
fi

# ---- install synth-city -----------------------------------------------------
info "Installing synth-city (editable)..."
if [[ "$WITH_DEV" == true ]]; then
    pip install -e "$SCRIPT_DIR[dev]" --quiet
else
    pip install -e "$SCRIPT_DIR" --quiet
fi

ok "synth-city installed"

# ---- .env -------------------------------------------------------------------
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    info "Creating .env from template..."
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    warn "Edit .env to add your API keys (CHUTES_API_KEY, etc.)"
else
    info ".env already exists — skipping"
fi

# ---- OpenClaw skill ---------------------------------------------------------
if [[ "$WITH_OPENCLAW" == true ]]; then
    info "Installing OpenClaw skill..."
    python "$SCRIPT_DIR/integrations/openclaw/setup.py"
fi

# ---- summary ----------------------------------------------------------------
echo ""
ok "Setup complete!"
echo ""
echo "  synth-city:       $SCRIPT_DIR"
echo "  open-synth-miner: $OSM_PATH"
if [[ "$SKIP_VENV" == false ]]; then
    echo "  virtualenv:       $VENV_DIR"
    echo ""
    echo "  Activate with:    source $VENV_DIR/bin/activate"
fi
echo ""
echo "  Quick start:"
echo "    python main.py sweep                      # baseline sweep"
echo "    python main.py pipeline                   # full agent pipeline"
echo "    python main.py bridge                     # start HTTP bridge"
echo ""
if [[ ! -f "$SCRIPT_DIR/.env" ]] || grep -q "your_chutes_api_key_here" "$SCRIPT_DIR/.env" 2>/dev/null; then
    warn "Don't forget to edit .env with your API keys!"
fi
