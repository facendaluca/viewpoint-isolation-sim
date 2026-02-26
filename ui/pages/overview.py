from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Overview")
    st.write(
        """
        This is the **Examiner Dashboard v1** scaffold.

        **Goal:** provide a clean UI entry point for running and reviewing experiments,
        while keeping all simulation logic outside the UI layer.
        """
    )

    st.subheader("What's in v1 (scaffold):")
    st.markdown(
        """
        - Clear navigation (4 pages)
        - Local launch (`streamlit run ui/Home.py`)
        - UI only modules (no simulation imports required yet)
        """
    )

    st.subheader("Next Steps")
    st.markdown(
        """
        - Shared app state (selected run, selected scenario, params)
        - Backend entrypoint placeholder: 'run_heuristic(config) -> run_dir'
        """
    )
