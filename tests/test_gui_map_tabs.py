"""Unit tests for GUI map tab descriptors/order."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from proc2d.app.gui.tabs import build_map_tab_specs, build_result_tab_labels


pytestmark = pytest.mark.unit


def test_build_map_tab_specs_has_structure_first_and_linecuts_enabled() -> None:
    specs = build_map_tab_specs()
    labels = [spec.label for spec in specs]
    assert labels[0] == "Structure"
    assert labels[1:] == [
        "Concentration",
        "Potential",
        "Electric Field",
        "Conductivity",
    ]
    assert all(spec.supports_linecut for spec in specs)


def test_build_result_tab_labels_places_structure_first() -> None:
    labels = build_result_tab_labels()
    assert labels[0] == "Structure"
    assert labels[:5] == [
        "Structure",
        "Concentration",
        "Potential",
        "Electric Field",
        "Conductivity",
    ]
    assert labels[-1] == "Storage"


def test_build_map_tab_specs_uses_additional_physical_maps_after_concentration() -> (
    None
):
    result = {
        "physical_maps": {
            "structure": object(),
            "concentration_cm3": object(),
            "potential_V": object(),
            "electric_field_mag_V_cm": object(),
            "conductivity_S_cm": object(),
            "custom_energy_eV": object(),
        }
    }

    specs = build_map_tab_specs(result)
    labels = [spec.label for spec in specs]
    assert labels == [
        "Structure",
        "Concentration",
        "Potential",
        "Electric Field",
        "Conductivity",
        "Custom Energy Ev",
    ]
    assert all(spec.supports_linecut for spec in specs)
