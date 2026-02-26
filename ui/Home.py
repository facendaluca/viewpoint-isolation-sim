from __future__ import annotations

import streamlit as st
from app_state import available_runs, available_scenarios, get_state, set_state
from pages import about, explore_results, overview, run_locally

APP_TITLE = "Examiner Dashboard v1"

_PAGE_RENDERS = {
    "Overview": overview.render,
    "Run Locally": run_locally.render,
    "Explore Results": explore_results.render,
    "About": about.render,
}


def _render_sidebar_state_controls() -> None:
    state = get_state(st.session_state)

    st.sidebar.subheader("App state")

    scenarios = available_scenarios()
    # Ensure current scenario is selectable even if configs changed
    if state.selected_scenario not in scenarios:
        scenarios = [state.selected_scenario, *scenarios]

    scenario = st.sidebar.selectbox(
        "Selected scenario",
        options=scenarios,
        index=scenarios.index(state.selected_scenario),
        help="Chosen config scenario for the run (UI-Only in v1)",
    )

    runs = available_runs()
    run_options = ["(none)", *runs]
    current = state.selected_run_dir or "(none)"
    if current not in run_options:
        run_options = [current, *run_options]

    selected = st.sidebar.selectbox(
        "Selected run directory",
        options=run_options,
        index=run_options.index(current),
        help="Picked run directory to explore (UI-only browse).",
    )

    new_state = state
    if scenario != state.selected_scenario:
        new_state = new_state.with_selected_scenario(scenario)

    run_dir_value = None if selected == "(none)" else selected
    if run_dir_value != state.selected_run_dir:
        new_state = new_state.with_selected_run_dir(run_dir_value)

    if new_state != state:
        set_state(st.session_state, new_state)

    st.sidebar.caption("State persists per browser session via Streamlit session_state.")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Scaffold only - UI contains **zero** simulation logic.")

    _render_sidebar_state_controls()

    page_name = st.sidebar.radio("Navigate", list(_PAGE_RENDERS.keys()), index=0)

    # Render the selected page
    _PAGE_RENDERS[page_name]()


if __name__ == "__main__":
    main()
