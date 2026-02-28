"""Streamlit GUI app assembly."""

from __future__ import annotations

from .forms import load_default_params, render_sidebar_form
from .session import get_last_result, push_recent_run, set_last_result
from .simulation import run_simulation
from .tabs import render_result_tabs


def run_gui() -> None:
    """Render Streamlit app."""
    import streamlit as st

    st.set_page_config(page_title="proc2d GUI", layout="wide")
    st.title("proc2d: Process 2D Cross-Section GUI")
    st.caption(
        "예제 deck 기본값을 로드해 파라미터를 조정하고, 같은 화면에서 "
        "맵/라인컷/메트릭/히스토리/비교를 확인할 수 있습니다."
    )

    defaults = load_default_params()
    try:
        submitted, params = render_sidebar_form(defaults)
    except ValueError as exc:
        st.error(f"Simulation failed: {exc}")
        submitted, params = False, None

    if submitted:
        try:
            if params is None:
                raise ValueError("Invalid form parameters")
            with st.spinner("Running simulation..."):
                result = run_simulation(params)
            set_last_result(result)
            push_recent_run(result, params)
            st.success(f"Simulation complete in {result['runtime_s']:.3f} s. Outputs: {result['outdir']}")
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    last_result = get_last_result()
    if last_result is None:
        st.info("왼쪽에서 파라미터를 조정하고 'Run Simulation'을 눌러 실행하세요.")
        return

    render_result_tabs(last_result)


__all__ = ["load_default_params", "run_gui", "run_simulation"]
