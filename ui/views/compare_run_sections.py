from __future__ import annotations

import streamlit as st

from fyp_sim.plotting.compare_run_data import CompareRunData


def run_selector(
    label: str,
    current: str | None,
    options: list[str],
    *,
    key: str,
) -> str | None:
    """Render a selectbox for a single run slot."""
    current_val = current or "(none)"
    if current_val not in options:
        options = [current_val, *options]

    picked = st.selectbox(
        f"Run {label}",
        options=options,
        index=options.index(current_val),
        key=key,
    )
    return None if picked == "(none)" else picked


def render_key_deltas(a: CompareRunData, b: CompareRunData) -> None:
    """Render the key-deltas metric panel."""
    st.subheader("Key Deltas")

    final_vii_a = float(a.df["isolation_index"].iloc[-1])
    final_vii_b = float(b.df["isolation_index"].iloc[-1])

    ttfl_a = a.lock_in.time_to_first_lock_in
    ttfl_b = b.lock_in.time_to_first_lock_in

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Final VII (cumulative)",
            f"{final_vii_a:.3f}",
            delta=f"{final_vii_a - final_vii_b:+.3f} vs B",
            delta_color="inverse",
        )

    with cols[1]:
        ttfl_a_str = "Never" if ttfl_a == -1 else str(ttfl_a)
        ttfl_b_str = "Never" if ttfl_b == -1 else str(ttfl_b)
        st.metric("Time to 1st lock-in (A)", ttfl_a_str, delta=f"B: {ttfl_b_str}")

    with cols[2]:
        tls_a = a.lock_in.total_lock_in_steps
        tls_b = b.lock_in.total_lock_in_steps
        st.metric(
            "Total lock-in steps",
            str(tls_a),
            delta=f"{tls_a - tls_b:+d} vs B",
            delta_color="inverse",
        )

    with cols[3]:
        watch_a = a.action_dist.proportions.get("Watch", 0.0)
        watch_b = b.action_dist.proportions.get("Watch", 0.0)
        st.metric(
            "Watch rate",
            f"{watch_a:.1%}",
            delta=f"{watch_a - watch_b:+.1%} vs B",
        )

    st.divider()

    detail_cols = st.columns(4)
    with detail_cols[0]:
        st.caption(f"Run A: {a.display_path}")
    with detail_cols[1]:
        st.caption(f"Seed A: {a.primary_seed}")
    with detail_cols[2]:
        st.caption(f"Run B: {b.display_path}")
    with detail_cols[3]:
        st.caption(f"Seed B: {b.primary_seed}")
