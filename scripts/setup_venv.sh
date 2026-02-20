#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[proc2d] venv install mode"

if ! python3 -m venv .venv >/dev/null 2>&1; then
  echo "[error] Failed to create .venv with python3 -m venv"
  echo "[hint] Your environment may be missing venv support (e.g. python3-venv package)."
  echo "       Install venv support in your OS and run again, or use scripts/setup_user_install.sh"
  exit 2
fi

source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev,gui]"

echo
echo "[proc2d] Install completed inside .venv"
echo "Next commands:"
echo "  source .venv/bin/activate"
echo "  python -m pytest"
echo "  python -m proc2d run examples/deck_basic.yaml --out outputs/run1"
