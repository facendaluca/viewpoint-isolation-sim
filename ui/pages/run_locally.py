from __future__ import annotations

import json

import streamlit as st
from app_state import get_state


def render() -> None:
    st.header("Run locally")
    st.caption("Scaffold only - no simulation is extended from the UI yet")

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

    try:
        loaded = json.loads(params_text)
        if not isinstance(loaded, dict):
            raise ValueError("Params JSON must be an object (dictionary).")
        st.success("Params parsed successfully.")
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid params: {e}")

    st.divider()

    st.subheader("Run (config coming in Milestone 3)")
    st.button("Run heuristic (placeholder)", disabled=True)
    st.info("Milestone 3 will wire this to a backend function outside the UI layer.")
