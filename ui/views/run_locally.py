from __future__ import annotations

import json

import streamlit as st

from fyp_sim.examiner_dashboard.backend import run_heuristic
from fyp_sim.examiner_dashboard.configs import build_resolved_config
from ui.app_state import get_state, set_state, to_display_path


def render() -> None:
    st.header("Run locally")
    st.caption("UI-only page - execution happens in the backend entrypoint in src/.")

    last = st.session_state.pop("dashboard_last_created_run", None)
    if isinstance(last, str) and last:
        st.success(f"Created run: `{last}`")

        col_a, col_b = st.columns([1, 3], vertical_alignment="center")
        with col_a:
            if st.button("Open Explore Results"):
                st.switch_page("pages/3_Explore_Results.py")
        with col_b:
            st.info("Go to **Explore Results** to browse the run directory.")

    state = get_state(st.session_state)

    st.subheader(f"**Selected scenario:** `{state.selected_scenario}`")
    st.write(
        "Run heuristic always creates a NEW run based on Selected scenario + params."
        "Selected run directory is for browsing only."
    )

    default_params = {"steps": 150, "top_k": 5, "seed": 0}
    params = state.params or default_params

    params_text = st.text_area(
        "Params JSON (stored in app state)",
        value=json.dumps(params, indent=2),
        height=220,
    )

    parsed_ok = False
    overrides: dict[str, object] = {}
    try:
        loaded = json.loads(params_text)
        if not isinstance(loaded, dict):
            raise ValueError("Params JSON must be an object (dictionary).")
        overrides = loaded
        st.success("Params parsed successfully.")
        parsed_ok = True

        # Keep shared state in sync
        if loaded != state.params:
            set_state(st.session_state, state.with_params(loaded))
            state = get_state(st.session_state)

    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid params: {e}")

    # Resolve scenario config + overrides
    resolved_ok = False
    resolved_cfg: dict[str, object] | None = None
    cfg_path_str: str | None = None

    if parsed_ok:
        try:
            resolved, cfg_path = build_resolved_config(state.selected_scenario, overrides)
            resolved_cfg = resolved
            cfg_path_str = str(cfg_path)
            resolved_ok = True
        except FileNotFoundError:
            st.error(
                f"Config file not found for scenario `{state.selected_scenario}`."
                "Expected it under `configs/`."
            )
        except Exception as e:
            st.error(f"Failed to build resolved config: {e}")

    with st.expander("Resolved config preview", expanded=True):
        if not parsed_ok:
            st.info("Fix Params JSON to preview the resolved config.")
        elif not resolved_ok:
            st.info("Resolved config unavailable (see error above).")
        else:
            st.caption(f"Loaded from: `{cfg_path_str}`")
            st.json(resolved_cfg)

    st.divider()

    st.subheader("Run (backend placeholder)")
    st.caption(
        "Creates a new run directory using the project's artefact convention and writes config_resolved.json + manifest.json."
    )

    col1, col2 = st.columns([1, 2], vertical_alignment="center")

    with col1:
        run_clicked = st.button("Run heuristic", disabled=not resolved_ok)

    with col2:
        st.write(f"**Selected run dir:** `{state.selected_run_dir or '(none)'}`")

    if run_clicked:
        if not resolved_ok or resolved_cfg is None:
            st.error("Resolved config is unavailable; cannot run.")
            return

        try:
            run_dir = run_heuristic(resolved_cfg, cfg_path=cfg_path_str)
        except Exception as e:  # Keep UI resilient; backend raises real errors
            st.error(f"Run failed: {e}")
            return

        created = to_display_path(run_dir)

        # Persist selected run for Explore Results page
        new_state = get_state(st.session_state).with_selected_run_dir(created)
        set_state(st.session_state, new_state)

        # Persist message across rerun so sidebar updates immediately
        st.session_state["dashboard_last_created_run"] = created
        st.rerun()
