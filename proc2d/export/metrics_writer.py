"""Metrics report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _ensure_outdir(outdir: str | Path) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _flatten_metrics(prefix: str, value: object, rows: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(name, sub, rows)
        return
    if isinstance(value, list):
        rows.append((prefix, json.dumps(value, ensure_ascii=False)))
        return
    rows.append((prefix, f"{value}"))


def save_metrics_json(metrics: dict, outdir: str | Path, filename: str = "metrics.json") -> Path:
    """Save analysis metrics in JSON format."""
    path = _ensure_outdir(outdir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_metrics_csv(metrics: dict, outdir: str | Path, filename: str = "metrics.csv") -> Path:
    """Save analysis metrics in flattened key-value CSV format."""
    path = _ensure_outdir(outdir) / filename
    rows: list[tuple[str, str]] = []
    _flatten_metrics("", metrics, rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])
    return path
