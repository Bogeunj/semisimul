"""Tests for GUI deck builder default process-step behavior."""

from __future__ import annotations

import pytest

from proc2d.config import GuiRunConfig, build_deck_from_gui_config


pytestmark = pytest.mark.unit


def _step_types(deck: dict[str, object]) -> list[str]:
    steps = deck.get("steps", [])
    assert isinstance(steps, list)
    out: list[str] = []
    for step in steps:
        assert isinstance(step, dict)
        out.append(str(step.get("type", "")))
    return out


def test_build_deck_from_gui_config_includes_all_core_process_steps_by_default() -> (
    None
):
    deck = build_deck_from_gui_config(GuiRunConfig())
    types = _step_types(deck)
    assert types[:7] == [
        "mask",
        "oxidation",
        "deposition",
        "implant",
        "anneal",
        "etch",
        "electrical",
    ]


def test_build_deck_from_gui_config_ignores_optional_enable_flags() -> None:
    cfg = GuiRunConfig.from_mapping(
        {
            "oxidation_enable": False,
            "deposition_enable": False,
            "etch_enable": False,
            "electrical_enable": False,
        }
    )
    deck = build_deck_from_gui_config(cfg)
    types = _step_types(deck)
    assert "oxidation" in types
    assert "deposition" in types
    assert "etch" in types
    assert "electrical" in types
