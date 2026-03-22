from __future__ import annotations

import streamlit as st

from fyp_sim.examiner_dashboard.configs import build_resolved_config
from ui.app_state import get_state, set_state, to_display_path
from ui.progress import StreamlitProgress
from ui.run_execution import ExecutionMode, execute_run
from ui.run_integrity import check_run_outputs
from ui.run_local_sections import (
    render_advanced_json_section,
    render_bounded_form,
    render_preset_selector,
    render_resolved_config_preview,
    render_run_action_panel,
)
from ui.run_local_state import init_page_state, sync_form_to_app_state


def render() -> None:
    st.header("Run locally")
    st.caption("UI-only page - execution happens in the backend entrypoint in src/.")

    # Show success notification if run was just created
    last = st.session_state.pop("dashboard_last_created_run", None)
    if isinstance(last, str) and last:
        st.success(f"Created run: `{last}`")

        col_a, col_b = st.columns([1, 3], vertical_alignment="center")
        with col_a:
            st.page_link("pages/3_Explore_Results.py", label="Open Explore Results", icon="🔍")
        with col_b:
            st.info("Go to **Explore Results** to browse the run directory.")

    init_page_state()
    state = get_state(st.session_state)

    mode_label = st.radio(
        "Execution mode",
        options=["Placeholder (fast)", "Real simulation (writes real logs)"],
        index=0,
        horizontal=True,
    )
    exec_mode = (
        ExecutionMode.PLACEHOLDER if mode_label == "Placeholder (fast)" else ExecutionMode.REAL
    )

    if exec_mode == ExecutionMode.REAL:
        st.info(
            "Real simulation runs in **heuristic mode only** in the hosted Examiner Dashboard."
            "LLM mode is supported only when running the repo locally (see README.md)."
        )

    st.divider()

    # 1. Preset Selector
    render_preset_selector()

    # 2. Form Input
    scenario_input, new_form_params = render_bounded_form(state.selected_scenario)

    # 3. Advanced JSON
    advanced_json = render_advanced_json_section()

    # 4. Sync and Validate
    parsed_ok, error_msg = sync_form_to_app_state(scenario_input, new_form_params, advanced_json)

    if not parsed_ok:
        st.error(f"Validation failed: {error_msg}")

    # Re-fetch state after sync
    state = get_state(st.session_state)

    resolved_ok = False
    resolved_cfg = None
    cfg_path_str = None

    if parsed_ok:
        try:
            resolved, cfg_path = build_resolved_config(state.selected_scenario, state.params)
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

    render_resolved_config_preview(parsed_ok, resolved_ok, resolved_cfg, cfg_path_str)

    run_clicked = render_run_action_panel(
        ready_to_run=resolved_ok,
        selected_run_dir=state.selected_run_dir,
    )

    if run_clicked:
        if not resolved_ok or resolved_cfg is None:
            st.error("Resolved config is unavailable; cannot run.")
            return

        progress = StreamlitProgress() if exec_mode == ExecutionMode.REAL else None

        try:
            result = execute_run(
                resolved_cfg=resolved_cfg,
                cfg_path=cfg_path_str,
                mode=exec_mode,
                progress=progress,
            )
        except Exception as e:
            st.error(f"Run failed: {e}")
            return

        # Integrity check
        if exec_mode == ExecutionMode.REAL:
            issues = check_run_outputs(result.run_dir)
            if issues:
                st.warning("Run completed, but some outputs look incomplete:")
                for iss in issues[:8]:
                    st.write(f"- **{iss.kind}**: `{iss.path}` - {iss.message}")
                if len(issues) > 8:
                    st.caption(f"Showing 8 of {len(issues)} issues.")

        created = to_display_path(result.run_dir)

        # Persist selected run for Explore Results page
        new_state = get_state(st.session_state).with_selected_run_dir(created)
        set_state(st.session_state, new_state)

        # Persist message across rerun so sidebar updates immediately
        st.session_state["dashboard_last_created_run"] = created
        st.rerun()
