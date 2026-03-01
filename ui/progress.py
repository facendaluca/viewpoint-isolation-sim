from __future__ import annotations

import streamlit as st

from ui.run_execution import ProgressSink


class StreamlitProgress(ProgressSink):
    """Minimal Streamlit progress implementation."""

    def __init__(self) -> None:
        self._bar = st.progress(0)
        self._status = st.empty()

    def on_seed_start(self, *, i: int, n: int, seed: int) -> None:
        self._status.write(f"Running seed {i}/{n} (seed={seed})")
        pct = int((i / max(n, 1)) * 100)
        self._bar.progress(min(max(pct, 0), 100))

    def on_done(self) -> None:
        self._bar.progress(100)
        self._status.write("Done!")
