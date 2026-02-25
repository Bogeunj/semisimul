from __future__ import annotations

from typing import Any


def make_default_gui_analyze_step(
    *,
    x_ref_um: float,
    y_ref_um: float,
    junction_thresholds_cm3: tuple[float, ...] = (1.0e17, 1.0e18),
    lateral_threshold_cm3: float = 1.0e17,
) -> dict[str, Any]:
    return {
        "type": "analyze",
        "junctions": [
            {
                "x_um": float(x_ref_um),
                "threshold_cm3": float(threshold),
            }
            for threshold in junction_thresholds_cm3
        ],
        "laterals": [
            {
                "y_um": float(y_ref_um),
                "threshold_cm3": float(lateral_threshold_cm3),
            }
        ],
        "sheet_dose": {"save_csv": True},
        "save": {"json": True, "csv": True},
    }
