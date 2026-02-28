"""Streamlit GUI components."""

from typing import Any

from .forms import load_default_params, render_sidebar_form


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Lazy proxy for GUI simulation adapter."""
    from .app import run_simulation as _impl

    return _impl(params)


def run_gui() -> None:
    """Lazy proxy for Streamlit app entrypoint."""
    from .app import run_gui as _impl

    _impl()

__all__ = ["load_default_params", "render_sidebar_form", "run_gui", "run_simulation"]
