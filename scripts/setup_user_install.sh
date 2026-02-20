#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[proc2d] User install mode (no venv)"
python3 -m pip install --user --break-system-packages -U pip setuptools wheel
python3 -m pip install --user --break-system-packages -e ".[dev,gui]"

echo
echo "[proc2d] Install completed. Recommended checks:"
echo "  python3 -m pytest"
echo "  python3 -m proc2d run examples/deck_basic.yaml --out outputs/run1"
echo "  python3 -m streamlit run proc2d/gui_script.py --server.port 8502"

case ":${PATH}:" in
  *":${HOME}/.local/bin:"*)
    ;;
  *)
    echo
    echo "[hint] ~/.local/bin is not on PATH."
    echo "Add this to your shell rc (~/.bashrc or ~/.zshrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
