"""
Placeholder for Compare Runs page.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from fyp_sim.plotting.compare_run_data import CompareRunData, load_compare_run
from fyp_sim.plotting.compare_run_plots import (
    plot_action_mix,
    plot_lockin_timeline,
    plot_vii_overlay,
)
from ui.app_state import available_runs, get_state, resolve_repo_path, set_state
from ui.views.compare_run_sections import render_key_deltas, run_selector


def _try_load_run(label: str, display_path: str) -> CompareRunData | None:
    """Attempt to load a run, showing errors in Streamlit on failure."""
    run_dir = resolve_repo_path(display_path)
    if not run_dir.is_dir():
        st.error(f"Run {label}: directory does not exist — {display_path}")
        return None

    try:
        return load_compare_run(label, display_path, run_dir)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        st.error(f"Run {label}: failed to load — {exc}")
        return None


def render() -> None:
    st.header("Compare Runs")
    st.caption("Select two runs to compare side-by-side.")

    state = get_state(st.session_state)

    runs = available_runs()
    if not runs:
        st.info("No runs found in outputs/runs/. Run a simulation first.")
        return

    options = ["(none)", *runs]

    col_a, col_b = st.columns(2)
    with col_a:
        picked_a = run_selector("A", state.selected_run_a, list(options), key="compare_sel_a")
    with col_b:
        picked_b = run_selector("B", state.selected_run_b, list(options), key="compare_sel_b")

    # Persist selections
    new_state = state
    if picked_a != state.selected_run_a:
        new_state = new_state.with_selected_run_a(picked_a)
    if picked_b != state.selected_run_b:
        new_state = new_state.with_selected_run_b(picked_b)
    if new_state != state:
        set_state(st.session_state, new_state)

    if not picked_a or not picked_b:
        st.info("Select two runs to compare.")
        return

    if picked_a == picked_b:
        st.info("Both selectors point to the same run. Select two different runs for comparison.")

    # Load both runs through the same backend pathway
    data_a = _try_load_run("A", picked_a)
    data_b = _try_load_run("B", picked_b)

    if data_a is None or data_b is None:
        return

    if data_a.params.threshold != data_b.params.threshold:
        st.warning(
            f"Runs use different thresholds: "
            f"A={data_a.params.threshold:.2f}, B={data_b.params.threshold:.2f}. "
            f"Comparison plots show each run's own threshold."
        )

    # Tabs — figure construction delegated to backend, display is UI-only
    tabs = st.tabs(["VII Trajectory", "Lock-in Timeline", "Action Mix", "Key Deltas"])

    with tabs[0]:
        fig = plot_vii_overlay(data_a, data_b)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        fig = plot_lockin_timeline(data_a, data_b)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        fig = plot_action_mix(data_a, data_b)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        render_key_deltas(data_a, data_b)
