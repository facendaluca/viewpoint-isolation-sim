from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("About")
    st.markdown(
        """
        **Examiner Dashboard v1** is a Streamlit UI scaffold for the Viewpoint Isolation Simulation project.

        Design principles:
        - UI layer is **presentation only**
        - Simulation logic stays in `src/` and is called via a minimal backend API
        - Deterministic simulation behaviour remains unchanged
        """
    )

    st.subheader("How to run")
    st.code("streamlit run ui/Home.py")
