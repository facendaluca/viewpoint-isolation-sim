from __future__ import annotations

import streamlit as st
from pages import about, explore_results, overview, run_locally

APP_TITLE = "Examiner Dashboard v1"

_PAGE_RENDERS = {
    "Overview": overview.render,
    "Run Locally": run_locally.render,
    "Explore Results": explore_results.render,
    "About": about.render,
}


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Scaffold only - UI contains **zero** simulation logic.")

    page_name = st.sidebar.radio("Navigate", list(_PAGE_RENDERS.keys()), index=0)

    # Render the selected page
    _PAGE_RENDERS[page_name]()


if __name__ == "__main__":
    main()
