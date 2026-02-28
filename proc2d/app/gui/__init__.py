"""Streamlit GUI components with lazy import proxies."""

from __future__ import annotations

from typing import Any


def load_default_params() -> dict[str, Any]:
    """Lazy proxy for GUI defaults loader."""
    from .forms import load_default_params as _impl

    return _impl()


def render_sidebar_form(defaults: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Lazy proxy for sidebar form renderer."""
    from .forms import render_sidebar_form as _impl

    return _impl(defaults)


def run_simulation(params: dict[str, Any]) -> dict[str, Any]:
    """Lazy proxy for GUI simulation adapter."""
    from .simulation import run_simulation as _impl

    return _impl(params)


def run_gui() -> None:
    """Lazy proxy for Streamlit app entrypoint."""
    from .app import run_gui as _impl

    _impl()


__all__ = ["load_default_params", "render_sidebar_form", "run_gui", "run_simulation"]
