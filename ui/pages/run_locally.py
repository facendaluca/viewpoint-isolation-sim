from __future__ import annotations

import json

import streamlit as st
from app_state import get_state, set_state, to_display_path

from fyp_sim.examiner_dashboard.backend import run_heuristic


def render() -> None:
    st.header("Run locally")
    st.caption("UI-only page - execution happens in the backend entrypoint in src/.")

    last = st.session_state.pop("dashboard_last_created_run", None)
    if isinstance(last, str) and last:
        st.success(f"Created run: `{last}`")
        st.info("Go to **Explore Results** to browse the run directory.")

    state = get_state(st.session_state)

    st.subheader(f"**Selected scenario:** `{state.selected_scenario}`")
    st.write("Edit params below - they are stored in shared app state.")

    default_params = {
        "steps": 150,
        "top_k": 5,
        "seed": 0,
    }
    params = state.params or default_params

    params_text = st.text_area(
        "Params JSON (stored in app state)",
        value=json.dumps(params, indent=2),
        height=220,
    )

    parsed_ok = False
    try:
        loaded = json.loads(params_text)
        if not isinstance(loaded, dict):
            raise ValueError("Params JSON must be an object (dictionary).")

        st.success("Params parsed successfully.")
        parsed_ok = True

    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid params: {e}")

    st.divider()

    st.subheader("Run (backend placeholder)")
    st.caption("Creates a new run directory and writes ")

    col1, col2 = st.columns([1, 2], vertical_alignment="center")

    with col1:
        run_clicked = st.button("Run heuristic", disabled=not parsed_ok)

    with col2:
        st.write(f"**Selected run dir:** `{state.selected_run_dir or '(none)'}`")

    if run_clicked:
        # Build a resolved config dict
        cfg = dict(state.params)
        cfg["scenario"] = state.selected_scenario

        try:
            run_dir = run_heuristic(cfg)
        except Exception as e:  # Keep UI resilient; backend raises real errors
            st.error(f"Run failed: {e}")
            return

        created = to_display_path(run_dir)

        # Persist selected run for Explore Results page
        new_state = get_state(st.session_state).with_selected_run_dir(created)
        set_state(st.session_state, new_state)

        # Persist message across rerun so sidebar updates immediately
        st.session_state["dashboard_last_created_run"] = created
        st.rerun()
