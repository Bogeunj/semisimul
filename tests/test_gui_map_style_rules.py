"""Unit tests for GUI map styling rules."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from proc2d.app.gui.tabs import map_style_for_spec_key, structure_material_legend


pytestmark = pytest.mark.unit


def test_structure_material_legend_has_descriptions() -> None:
    legend = structure_material_legend()
    assert {item[0] for item in legend} == {0, 1}
    assert all(item[2] for item in legend)


def test_concentration_style_has_gray_floor() -> None:
    style = map_style_for_spec_key("concentration")
    assert style.low_color == "#b3b3b3"


def test_potential_and_efield_use_blue_red_cmap() -> None:
    p_style = map_style_for_spec_key("potential")
    e_style = map_style_for_spec_key("electric_field")
    assert p_style.cmap_name == "bwr"
    assert e_style.cmap_name == "bwr"
