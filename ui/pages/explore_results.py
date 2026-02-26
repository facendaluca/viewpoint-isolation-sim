from __future__ import annotations

from pathlib import Path

import streamlit as st


def render() -> None:
    st.header("Explore Results")
    st.caption("Scaffold only - this page will later browse outputs without running simulations.")

    st.subheader("Run directory (placeholder)")
    run_dir = st.text_input("Path to run directory", value="outputs/runs/")

    p = Path(run_dir)
    if p.exists():
        st.success(f"Found: {p.resolve()}")
        if p.is_dir():
            # Lightweight display: top-level listing only
            items = sorted(p.iterdir())
            st.write(f"Items: {len(items)}")
            st.code("\n".join(i.name for i in items[:50]) or "(empty)")
            if len(items) > 50:
                st.caption("Showing first 50 entries.")
    else:
        st.warning("Path does not exist (yet).")
