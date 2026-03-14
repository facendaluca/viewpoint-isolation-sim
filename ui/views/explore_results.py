from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.app_state import available_runs, get_state, resolve_repo_path, set_state
from ui.figure_browser import list_plot_images, render_figure_browser
from ui.results_io import read_csv_head, read_json
from ui.run_inspector import (
    find_config_file,
    find_manifest_file,
    find_seed_log,
    find_summary_file,
    list_files,
    list_seed_dirs,
)


def render_overview(
    run_dir: Path,
    manifest_path: Path | None,
    config_path: Path | None,
    summary_path: Path | None,
    plots_dir: Path,
    seeds_dir: Path,
) -> None:
    st.subheader("Run overview")
    st.write(f"**Run dir:** `{run_dir}`")

    if manifest_path:
        try:
            manifest = read_json(manifest_path)
            st.write(f"**Mode:** `{manifest.get('mode')}`")
            st.write(f"**Run ID:** `{manifest.get('run_id')}`")
            st.write(f"**Date:** `{manifest.get('date_ymd')}`")
            st.write(f"**Seeds:** `{manifest.get('seeds')}`")
        except Exception as e:
            st.error(f"Failed to read manifest file: {e}")
    else:
        st.warning("Manifest file not found (this directory may not be a valid run).")

    plot_count = len(list_plot_images(plots_dir))

    st.subheader("Expected artefacts")
    cols = st.columns(3)
    cols[0].write(f"- config file: {'exists' if config_path else 'missing'}")
    cols[1].write(f"- manifest file: {'exists' if manifest_path else 'missing'}")
    cols[2].write(f"- summary.csv: {'exists' if summary_path else 'missing'}")
    st.write(
        f"- plots/: {'exists' if plots_dir.is_dir() else 'missing'}"
        f"{f' ({plot_count} image files)' if plots_dir.is_dir() else ''}"
    )
    st.write(f"- seeds/: {'exists' if seeds_dir.is_dir() else 'missing'}")


def render_config(config_path: Path | None) -> None:
    st.subheader("Config")
    if not config_path:
        st.info("No config file found.")
    else:
        try:
            cfg = read_json(config_path)
            st.json(cfg)
            st.download_button(
                f"Download {config_path.name}",
                data=config_path.read_bytes(),
                file_name=config_path.name,
                mime="application/json",
            )
        except Exception as e:
            st.error(f"Failed to read {config_path.name}: {e}")


def render_manifest(manifest_path: Path | None) -> None:
    st.subheader("Manifest")
    if not manifest_path:
        st.info("No manifest/meta file found.")
    else:
        try:
            manifest = read_json(manifest_path)
            st.json(manifest)
            st.download_button(
                f"Download {manifest_path.name}",
                data=manifest_path.read_bytes(),
                file_name=manifest_path.name,
                mime="application/json",
            )
        except Exception as e:
            st.error(f"Failed to read {manifest_path.name}: {e}")


def render_summary(summary_path: Path | None) -> None:
    st.subheader("Summary")
    if not summary_path:
        st.info("No summary.csv found.")
    else:
        try:
            df = read_csv_head(summary_path, n=200)
            if df.empty:
                st.info("summary.csv is empty.")
            else:
                st.dataframe(df, width="stretch")
            st.download_button(
                "Download summary.csv",
                data=summary_path.read_bytes(),
                file_name="summary.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Failed to read summary.csv: {e}")


def render_seeds(run_dir: Path) -> None:
    st.subheader("Seeds")
    seed_dirs = list_seed_dirs(run_dir)
    if not seed_dirs:
        st.info("No valid seed subdirectories found.")
        return

    seed_names = [p.name for p in seed_dirs]
    seed_name = st.selectbox("Select a seed", options=seed_names, index=0)
    seed_dir = next(p for p in seed_dirs if p.name == seed_name)
    run_log = find_seed_log(seed_dir)

    st.write(f"**Seed dir:** `{seed_dir.relative_to(run_dir.parent)}`")
    if not run_log:
        st.info("run_log.csv not found for this seed.")
    else:
        try:
            df = read_csv_head(run_log, n=200)
            if df.empty:
                st.info("run_log.csv exists but is empty (placeholder).")
            else:
                st.dataframe(df, width="stretch")
            st.download_button(
                "Download run_log.csv",
                data=run_log.read_bytes(),
                file_name="run_log.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Failed to read run_log.csv: {e}")


def render_files(run_dir: Path) -> None:
    st.subheader("Files")
    lines = list_files(run_dir)
    if not lines:
        st.info("No files found in this run directory.")
    else:
        st.code("\n".join(lines[:250]), language="text")
        if len(lines) > 250:
            st.caption("Showing first 250 lines.")


def render_debug_paths(
    run_dir: Path, config_path: Path | None, manifest_path: Path | None, summary_path: Path | None
) -> None:
    with st.expander("Debug paths"):
        st.write(f"**run_dir:** `{run_dir}`")
        st.write(f"**config:** `{config_path}`")
        st.write(f"**manifest:** `{manifest_path}`")
        st.write(f"**summary:** `{summary_path}`")


def render() -> None:
    st.header("Explore Results")
    st.caption("Browse existing run artefacts. No simulation logic here.")

    state = get_state(st.session_state)

    runs = available_runs()
    options = ["(none)", *runs]
    current = state.selected_run_dir or "(none)"
    if current not in options:
        options = [current, *options]

    picked = st.selectbox("Select a run directory", options=options, index=options.index(current))
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

    config_path = find_config_file(run_dir)
    manifest_path = find_manifest_file(run_dir)
    summary_path = find_summary_file(run_dir)
    plots_dir = run_dir / "plots"
    seeds_dir = run_dir / "seeds"

    render_debug_paths(run_dir, config_path, manifest_path, summary_path)

    tabs = st.tabs(["Overview", "Config", "Manifest", "Summary", "Seeds", "Plots", "Files"])

    with tabs[0]:
        render_overview(run_dir, manifest_path, config_path, summary_path, plots_dir, seeds_dir)

    with tabs[1]:
        render_config(config_path)

    with tabs[2]:
        render_manifest(manifest_path)

    with tabs[3]:
        render_summary(summary_path)

    with tabs[4]:
        render_seeds(run_dir)

    with tabs[5]:
        render_figure_browser(plots_dir)

    with tabs[6]:
        render_files(run_dir)
