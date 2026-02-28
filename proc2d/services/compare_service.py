"""Shared compare helpers for GUI and other adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


def scalarize_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metrics dictionary to scalar key/value map."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(scalarize_metrics(value, prefix=name))
        elif isinstance(value, (int, float, np.floating)):
            out[name] = float(value)
    return out


def build_metric_delta_rows(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> list[dict[str, str]]:
    """Build tabular metric delta rows from two metrics payloads."""
    scalar_a = scalarize_metrics(metrics_a)
    scalar_b = scalarize_metrics(metrics_b)
    rows: list[dict[str, str]] = []
    for key in sorted(set(scalar_a) & set(scalar_b)):
        a_val = scalar_a[key]
        b_val = scalar_b[key]
        rows.append(
            {
                "metric": key,
                "A": f"{a_val:.6g}",
                "B": f"{b_val:.6g}",
                "delta(B-A)": f"{(b_val - a_val):.6g}",
            }
        )
    return rows
