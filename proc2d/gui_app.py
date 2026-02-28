"""Compatibility wrapper for Streamlit GUI app."""

from __future__ import annotations

from typing import Any


def load_default_params() -> dict[str, Any]:
    """Load default GUI params lazily from app module."""
    from .app.gui.app import load_default_params as _impl

    return _impl()


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Run GUI simulation adapter lazily from app module."""
    from .app.gui.app import run_simulation as _impl

    return _impl(params)


def run_gui() -> None:
    """Run Streamlit GUI app lazily from app module."""
    from .app.gui.app import run_gui as _impl

    _impl()


__all__ = ["load_default_params", "run_gui", "run_simulation"]
