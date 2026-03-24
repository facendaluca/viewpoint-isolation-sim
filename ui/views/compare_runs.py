"""
Placeholder for Compare Runs page.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Compare Runs")
    st.caption("Select two runs to compare side-by-side.")

    col_a, col_b = st.columns(2)

    st.info("Select two runs to compare.")

    # Warn if thresholds differ
    st.warning(
        "Runs use different thresholds: placeholderComparison plots show each run's own threshold."
    )

    # Tabs
    tabs = st.tabs(["VII Trajectory", "Lock-in Timeline", "Action Mix", "Key Deltas"])

    with tabs[0]:
        st.write("VII Trajectory")
    with tabs[1]:
        st.write("Lock-in Timeline")
    with tabs[2]:
        st.write("Action Mix")
    with tabs[3]:
        st.write("Key Deltas")
