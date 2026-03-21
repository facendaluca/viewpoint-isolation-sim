from __future__ import annotations

from datetime import datetime

import streamlit as st

from fyp_sim.plotting.generation import PlotGenerationError, generate_plots_for_run, get_plot_status
from ui.figure_browser import render_figure_browser
from ui.run_context import RunContext


def _flash_key(ctx: RunContext) -> str:
    safe_dir = str(ctx.run_dir).replace("\\", "__").replace("/", "__").replace(":", "_")
    return f"explore_results::plot_generation::{safe_dir}"


def _store_success_flash(ctx: RunContext, plot_count: int) -> None:
    st.session_state[_flash_key(ctx)] = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "count": plot_count,
    }


def _render_success_flash(ctx: RunContext) -> None:
    payload = st.session_state.pop(_flash_key(ctx), None)
    if not isinstance(payload, dict):
        return

    timestamp = payload.get("timestamp")
    count = payload.get("count")
    if isinstance(timestamp, str) and isinstance(count, int):
        noun = "plot" if count == 1 else "plots"
        st.success(f"Generated {count} {noun} at {timestamp}.")


def _render_status_summary(ctx: RunContext) -> None:
    status = get_plot_status(ctx.run_dir)

    cols = st.columns(3)
    cols[0].write(f"**Plots status:** `{status.label}`")
    cols[1].write(f"**Images found:** `{len(status.plot_files)}`")
    cols[2].write(f"**plots/ directory:** `{'present' if ctx.plots_dir.is_dir() else 'missing'}`")

    if status.label == "Found":
        st.success("Plots are available for this run.")
    elif status.label == "Outdated":
        st.warning("Plots exist, but they appear older than the current run inputs.")
    else:
        st.info("No plot images have been generated for this run yet.")


def _render_generation_controls(ctx: RunContext) -> None:
    overwrite = st.checkbox(
        "Regenerate (overwrite)",
        value=False,
        help="Delete existing plot outputs in plots/ before regenerating them.",
    )

    button_label = "Regenerate plots" if overwrite else "Generate plots"

    if not st.button(button_label):
        return

    try:
        with st.spinner("Generating plots for the selected run..."):
            plot_files = generate_plots_for_run(ctx.run_dir, overwrite=overwrite)
    except PlotGenerationError as e:
        st.error(f"Plot generation failed: {e}")
        return
    except Exception:
        st.error(
            "Plot generation failed due to an unexpected error. No stack trace is shown in the UI."
        )
        return

    _store_success_flash(ctx, len(plot_files))
    st.rerun()


def render_plots_panel(ctx: RunContext) -> None:
    st.subheader("Generate plots")

    _render_success_flash(ctx)
    _render_status_summary(ctx)
    _render_generation_controls(ctx)

    st.divider()
    render_figure_browser(ctx.plots_dir)
