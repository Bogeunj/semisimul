"""Launcher for Streamlit-based proc2d GUI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Launch the Streamlit GUI app."""
    try:
        from streamlit.web import cli as stcli
    except Exception:
        print(
            "Streamlit is required for GUI. Install with:\n"
            "  python3 -m pip install --user --break-system-packages -e '.[gui]'"
        )
        return 2

    script = Path(__file__).resolve().with_name("gui_script.py")
    sys.argv = ["streamlit", "run", str(script)]
    return stcli.main()
