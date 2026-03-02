"""Unit tests for GUI number parsing with optional unit suffixes."""

from __future__ import annotations

import pytest

from proc2d.app.gui.unit_input import parse_number_with_optional_unit


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "unit", "expected"),
    [
        ("1e15", "cm^-3", 1.0e15),
        ("1e15 [cm^-3]", "cm^-3", 1.0e15),
        ("1e15 cm^-3", "cm^-3", 1.0e15),
        ("1,000,000 [cm^-3]", "cm^-3", 1.0e6),
        ("2.5e-3 [cm^2/s]", "cm^2/s", 2.5e-3),
    ],
)
def test_parse_number_with_optional_unit_accepts_plain_or_unit_text(
    raw: str, unit: str, expected: float
) -> None:
    value = parse_number_with_optional_unit(raw, expected_unit=unit, field_name="x")
    assert value == pytest.approx(expected)


def test_parse_number_with_optional_unit_rejects_wrong_unit() -> None:
    with pytest.raises(ValueError, match=r"must use unit"):
        parse_number_with_optional_unit(
            "1e15 [m^-3]", expected_unit="cm^-3", field_name="bg"
        )


def test_parse_number_with_optional_unit_rejects_invalid_text() -> None:
    with pytest.raises(ValueError, match=r"Invalid numeric value"):
        parse_number_with_optional_unit(
            "not-a-number", expected_unit="cm^-3", field_name="bg"
        )
