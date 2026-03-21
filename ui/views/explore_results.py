from __future__ import annotations

import streamlit as st

from ui.app_state import available_runs, get_state, resolve_repo_path, set_state
from ui.run_context import RunContext
from ui.views.explore_results_sections import (
    render_config,
    render_debug_paths,
    render_files,
    render_manifest,
    render_overview,
    render_seeds,
    render_summary,
)
from ui.views.run_plots_panel import render_plots_panel


def render() -> None:
    st.header("Explore Results")
    st.caption("Browse existing run artefacts. No simulation logic here.")

    state = get_state(st.session_state)

    runs = available_runs()
    options = ["(none)", *runs]
    current = state.selected_run_dir or "(none)"
    if current not in options:
        options = [current, *options]

    picked = st.selectbox(
        "Select a run directory",
        options=options,
        index=options.index(current),
    )
    picked_dir = None if picked == "(none)" else picked

    if picked_dir != state.selected_run_dir:
        set_state(st.session_state, state.with_selected_run_dir(picked_dir))
        state = get_state(st.session_state)

    if not state.selected_run_dir:
        st.info("Select a run directory to inspect.")
        return

    run_dir = resolve_repo_path(state.selected_run_dir)
    if not run_dir.is_dir():
        st.error("Selected run directory does not exist or is not a directory.")
        return

    ctx = RunContext.from_run_dir(run_dir)

    render_debug_paths(
        ctx.run_dir,
        ctx.config_path,
        ctx.manifest_path,
        ctx.summary_path,
    )

    tabs = st.tabs(["Overview", "Config", "Manifest", "Summary", "Seeds", "Plots", "Files"])

    with tabs[0]:
        render_overview(
            ctx.run_dir,
            ctx.manifest_path,
            ctx.config_path,
            ctx.summary_path,
            ctx.plots_dir,
            ctx.seeds_dir,
        )

    with tabs[1]:
        render_config(ctx.config_path)

    with tabs[2]:
        render_manifest(ctx.manifest_path)

    with tabs[3]:
        render_summary(ctx.summary_path)

    with tabs[4]:
        render_seeds(ctx.run_dir)

    with tabs[5]:
        render_plots_panel(ctx)

    with tabs[6]:
        render_files(ctx.run_dir)
