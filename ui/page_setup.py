from __future__ import annotations

import streamlit as st

from ui.sidebar_state import render_app_state_controls


def setup_page(title: str) -> None:
    # Must be the first Streamlit call in each page script
    st.set_page_config(
        page_title=title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Examiner Dashboard v1")
    st.sidebar.caption(
        "Examiner-facing dashboard for generating heuristic runs, reviewing artefacts, and comparing results."
    )

    render_app_state_controls()
