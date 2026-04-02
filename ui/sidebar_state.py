from __future__ import annotations

import streamlit as st

from ui.app_state import available_runs, get_state, set_state


def render_app_state_controls() -> None:
    """Render the shared Scenario + Selected run directory controls in the sidebar."""
    state = get_state(st.session_state)

    st.sidebar.subheader("App state")

    st.sidebar.markdown("**Selected scenario**")
    st.sidebar.code(state.selected_scenario or "(none)")

    runs = available_runs()
    run_options = ["(none)", *runs]
    current_run = state.selected_run_dir or "(none)"
    if current_run not in run_options:
        run_options = [current_run, *run_options]

    selected_run = st.sidebar.selectbox(
        "Selected run directory",
        options=run_options,
        index=run_options.index(current_run),
    )

    selected_run_dir = None if selected_run == "(none)" else selected_run

    new_state = state
    if selected_run_dir != state.selected_run_dir:
        new_state = new_state.with_selected_run_dir(selected_run_dir)

    if new_state != state:
        set_state(st.session_state, new_state)

    st.sidebar.caption("State persists per browser session via Streamlit session_state.")
