from __future__ import annotations

import json

import streamlit as st


def render() -> None:
    st.header("Run locally")
    st.caption("Scaffold only - no simulation is extended from the UI yet")

    st.subheader("Config (placeholder)")
    st.write("In v1, this is just a placeholder dict editor to prove the UI flow")

    default_config = {
        "experiment": "baseline",
        "steps": 150,
        "top_k": 5,
        "seed": 0,
    }

    config_text = st.text_area(
        "Config JSON",
        value=json.dumps(default_config, indent=2),
        height=220,
    )

    # Parse-only validation (still no experiment logic)
    try:
        _ = json.loads(config_text)
        st.success("Config parsed successfully")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")

    st.divider()

    st.subheader("Run (config coming in Milestone 3)")
    st.button("Run heuristic (placeholder)", disabled=True)
    st.info("Milestone 3 will wire this to a backend function outside the UI layer.")
