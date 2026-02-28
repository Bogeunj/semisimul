"""Compatibility tests for legacy wrapper modules."""

from __future__ import annotations

import pytest

import proc2d.app.cli as app_cli
import proc2d.cli as compat_cli


pytestmark = pytest.mark.adapter


def test_cli_wrapper_exports_app_entrypoints() -> None:
    assert compat_cli.main is app_cli.main
    assert compat_cli.build_parser is app_cli.build_parser


def test_gui_wrapper_import_is_lazy() -> None:
    import proc2d.gui_app as compat_gui

    assert callable(compat_gui.run_gui)
    assert callable(compat_gui.run_simulation)
    assert callable(compat_gui.load_default_params)


def test_gui_wrapper_smoke_with_streamlit() -> None:
    streamlit = pytest.importorskip("streamlit")
    assert streamlit is not None

    from proc2d.gui_app import load_default_params

    defaults = load_default_params()
    assert isinstance(defaults, dict)
    assert "Lx_um" in defaults
