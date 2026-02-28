"""Compare tab renderer for last two GUI runs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .session import COMPARE_LINECUT_KEY, get_recent_runs


def _scalarize_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_scalarize_metrics(value, prefix=name))
        elif isinstance(value, (int, float, np.floating)):
            out[name] = float(value)
    return out


def render_compare_tab() -> None:
    """Render side-by-side compare for the two latest runs."""
    runs = get_recent_runs()
    if len(runs) < 2:
        st.info("비교하려면 최소 2회 실행이 필요합니다.")
        return

    run_a = runs[-2]
    run_b = runs[-1]

    st.write(f"A: `{run_a['run_id']}` ({run_a['timestamp']})")
    st.write(f"B: `{run_b['run_id']}` ({run_b['timestamp']})")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Run A map")
        if run_a.get("C") is not None and run_b.get("C") is not None:
            C_a = np.asarray(run_a["C"], dtype=float)
            C_b = np.asarray(run_b["C"], dtype=float)
            vmin = min(float(np.min(C_a)), float(np.min(C_b)))
            vmax = max(float(np.max(C_a)), float(np.max(C_b)))
            fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=130)
            ax.imshow(C_a, origin="upper", aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title("A (same scale)")
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
        elif run_a.get("png_bytes"):
            st.image(run_a["png_bytes"], use_container_width=True)
        else:
            st.info("Run A image unavailable (PNG 미저장 + full C 미보관)")

    with col2:
        st.caption("Run B map")
        if run_a.get("C") is not None and run_b.get("C") is not None:
            C_a = np.asarray(run_a["C"], dtype=float)
            C_b = np.asarray(run_b["C"], dtype=float)
            vmin = min(float(np.min(C_a)), float(np.min(C_b)))
            vmax = max(float(np.max(C_a)), float(np.max(C_b)))
            fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=130)
            ax.imshow(C_b, origin="upper", aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title("B (same scale)")
            ax.set_axis_off()
            st.pyplot(fig)
            plt.close(fig)
        elif run_b.get("png_bytes"):
            st.image(run_b["png_bytes"], use_container_width=True)
        else:
            st.info("Run B image unavailable (PNG 미저장 + full C 미보관)")

    linecut_kind = st.radio(
        "Compare linecut",
        ["Vertical", "Horizontal"],
        horizontal=True,
        key=COMPARE_LINECUT_KEY,
    )
    if linecut_kind == "Vertical":
        a_x = np.asarray(run_a["linecuts"]["vertical"]["coord_um"], dtype=float)
        a_y = np.asarray(run_a["linecuts"]["vertical"]["values_cm3"], dtype=float)
        b_x = np.asarray(run_b["linecuts"]["vertical"]["coord_um"], dtype=float)
        b_y = np.asarray(run_b["linecuts"]["vertical"]["values_cm3"], dtype=float)
        x_label = "y [um]"
    else:
        a_x = np.asarray(run_a["linecuts"]["horizontal"]["coord_um"], dtype=float)
        a_y = np.asarray(run_a["linecuts"]["horizontal"]["values_cm3"], dtype=float)
        b_x = np.asarray(run_b["linecuts"]["horizontal"]["coord_um"], dtype=float)
        b_y = np.asarray(run_b["linecuts"]["horizontal"]["values_cm3"], dtype=float)
        x_label = "x [um]"

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=140)
    ax.plot(a_x, a_y, lw=1.8, label=f"A ({run_a['run_id']})")
    ax.plot(b_x, b_y, lw=1.8, label=f"B ({run_b['run_id']})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("C [cm^-3]")
    ax.set_title("Linecut overlay")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    metrics_a = run_a.get("metrics") or {}
    metrics_b = run_b.get("metrics") or {}
    if metrics_a and metrics_b:
        scalar_a = _scalarize_metrics(metrics_a)
        scalar_b = _scalarize_metrics(metrics_b)
        rows: list[dict[str, str]] = []
        for key in sorted(set(scalar_a) & set(scalar_b)):
            a_val = scalar_a[key]
            b_val = scalar_b[key]
            rows.append({
                "metric": key,
                "A": f"{a_val:.6g}",
                "B": f"{b_val:.6g}",
                "delta(B-A)": f"{(b_val - a_val):.6g}",
            })
        st.subheader("Metrics difference")
        st.table(rows)
