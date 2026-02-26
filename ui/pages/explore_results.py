from __future__ import annotations

from pathlib import Path

import streamlit as st
from app_state import available_runs, get_state, set_state


def render() -> None:
    st.header("Explore Results")
    st.caption("Scaffold only - this page will later browse outputs without running simulations.")

    state = get_state(st.session_state)
    st.write(f"**Selected run directory:** `{state.selected_run_dir or "'(none)'"}`")

    st.subheader("Pick a run directory")
    runs = available_runs()
    options = ["(none)", *runs]
    current = state.selected_run_dir or "(None)"
    if current not in options:
        options = [current, *options]

    picked = st.selectbox("Runs", options=options, index=options.index(current))
    picked_dir = None if picked == "(none)" else picked

    if picked_dir != state.selected_run_dir:
        set_state(st.session_state, state.with_selected_run_dir(picked_dir))
        state = get_state(st.session_state)

    st.divider()

    if not state.selected_run_dir:
        st.warning("Selected path does not exist.")
        return

    p = Path(state.selected_run_dir)
    if not p.exists():
        st.warning("Selected path does not exist.")
        return
    if not p.is_dir():
        st.warning("Selected path is not a directory.")
        return

    items = sorted(p.iterdir(), key=lambda x: x.name)
    st.write(f"Items in `{p}`: {len(items[:80]) or '(empty)'}")
    if len(items) > 80:
        st.caption("Showing first 80 entries.")
