"""Unit tests for deck parser helpers."""

from __future__ import annotations

import pytest

from proc2d.config.parser import parse_domain_config, parse_step_configs, parse_steps


pytestmark = pytest.mark.unit


def test_parse_domain_config_prefers_top_level_background_doping() -> None:
    deck = {
        "domain": {
            "Lx_um": 1.0,
            "Ly_um": 0.4,
            "Nx": 31,
            "Ny": 21,
            "background_doping_cm3": 2.0e15,
        },
        "background_doping_cm3": 7.5e14,
    }

    domain = parse_domain_config(deck)
    assert domain.background_doping_cm3 == 7.5e14


@pytest.mark.parametrize("steps", [[], None, "not-a-list"])
def test_parse_steps_rejects_empty_or_non_list(steps: object) -> None:
    with pytest.raises(ValueError, match=r"deck\.steps must be a non-empty list"):
        parse_steps({"steps": steps})


def test_parse_step_configs_rejects_invalid_oxidation_apply_on() -> None:
    deck = {
        "steps": [
            {
                "type": "oxidation",
                "time_s": 1.0,
                "A_um": 0.1,
                "B_um2_s": 0.01,
                "apply_on": "sidewall",
            }
        ]
    }

    with pytest.raises(ValueError, match=r"steps\[0\]\.apply_on must be one of"):
        parse_step_configs(deck)
