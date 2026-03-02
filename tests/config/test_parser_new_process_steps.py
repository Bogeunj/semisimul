"""Parser tests for newly added process/electrical steps."""

from __future__ import annotations

import pytest

from proc2d.config.parser import parse_step_configs


pytestmark = pytest.mark.unit


def test_parse_step_configs_accepts_deposition_step() -> None:
    typed = parse_step_configs(
        {"steps": [{"type": "deposition", "thickness_um": 0.02}]}
    )
    assert len(typed) == 1
    assert typed[0].type == "deposition"


def test_parse_step_configs_accepts_etch_step() -> None:
    typed = parse_step_configs({"steps": [{"type": "etch", "thickness_um": 0.01}]})
    assert len(typed) == 1
    assert typed[0].type == "etch"


def test_parse_step_configs_accepts_electrical_step() -> None:
    typed = parse_step_configs({"steps": [{"type": "electrical"}]})
    assert len(typed) == 1
    assert typed[0].type == "electrical"


def test_parse_step_configs_rejects_negative_deposition_thickness() -> None:
    with pytest.raises(ValueError, match=r"thickness_um"):
        parse_step_configs({"steps": [{"type": "deposition", "thickness_um": -0.01}]})


def test_parse_step_configs_rejects_negative_etch_thickness() -> None:
    with pytest.raises(ValueError, match=r"thickness_um"):
        parse_step_configs({"steps": [{"type": "etch", "thickness_um": -0.01}]})


def test_parse_step_configs_lists_new_step_types_for_unknown_type() -> None:
    with pytest.raises(ValueError, match=r"deposition, etch, electrical"):
        parse_step_configs({"steps": [{"type": "unknown"}]})
