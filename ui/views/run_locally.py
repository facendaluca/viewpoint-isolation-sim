from __future__ import annotations

import streamlit as st

from fyp_sim.examiner_dashboard.configs import build_resolved_config
from ui.app_state import get_state, set_state, to_display_path
from ui.progress import StreamlitProgress
from ui.run_execution import ExecutionMode, execute_run
from ui.run_integrity import check_run_outputs
from ui.run_local_catalog import get_preset
from ui.run_local_form import RunLocalFormInput, build_submission, params_from_raw
from ui.run_local_sections import (
    render_advanced_json_section,
    render_bounded_form,
    render_execution_mode,
    render_failed_state,
    render_preset_selector,
    render_resolved_config_preview,
    render_run_action_panel,
    render_run_feedback,
    render_success_banner,
    render_validation_errors,
)
from ui.run_local_state import (
    get_current_advanced_json,
    get_current_preset_id,
    handle_preset_change,
    init_page_state,
    sync_valid_build,
)


def render() -> None:
    # 1. Header & success banner
    st.header("Run locally")
    st.caption("UI-only page - execution happens in the backend entrypoint in src/.")

    last = st.session_state.pop("dashboard_last_created_run", None)
    if isinstance(last, str) and last:
        render_success_banner(last)

    # 2. Init & execution mode
    init_page_state()
    state = get_state(st.session_state)

    mode_label = render_execution_mode()
    exec_mode = (
        ExecutionMode.PLACEHOLDER if mode_label == "Placeholder (fast)" else ExecutionMode.REAL
    )

    st.divider()

    # 3. Preset selector — rerun on change
    current_preset_id = get_current_preset_id()
    current_preset = get_preset(current_preset_id)
    st.subheader(f"**Selected scenario:** `{current_preset.scenario}`")

    new_preset_id = render_preset_selector(current_preset_id)
    if new_preset_id:
        handle_preset_change(new_preset_id)
        st.rerun()

    # 4. Bounded controls & advanced JSON
    current_params = params_from_raw(state.params, defaults=current_preset.defaults)
    new_params = render_bounded_form(current_params)
    advanced_json = render_advanced_json_section(get_current_advanced_json())

    # 5. Build & validate submission
    build_result = build_submission(
        RunLocalFormInput(
            preset_id=current_preset_id, params=new_params, advanced_json=advanced_json
        )
    )
    render_validation_errors(build_result.errors)

    # 6. Resolve config (gated on valid submission)
    resolved_ok = False
    resolved_cfg = None
    cfg_path_str = None

    if not build_result.errors:
        sync_valid_build(build_result.preset.scenario, build_result.overrides, advanced_json)
        try:
            resolved, cfg_path = build_resolved_config(
                build_result.preset.scenario, build_result.overrides
            )
            resolved_cfg, cfg_path_str, resolved_ok = resolved, str(cfg_path), True
        except FileNotFoundError:
            st.error(
                f"Config file not found for scenario `{build_result.preset.scenario}`. "
                "Expected it under `configs/`."
            )
        except Exception as e:
            st.error(f"Failed to build resolved config: {e}")

    render_resolved_config_preview(not build_result.errors, resolved_cfg, cfg_path_str)

    # 7. Run
    run_clicked = render_run_action_panel(resolved_ok, state.selected_run_dir, exec_mode)

    if run_clicked and resolved_ok and resolved_cfg:
        progress = StreamlitProgress() if exec_mode == ExecutionMode.REAL else None

        if exec_mode == ExecutionMode.PLACEHOLDER:
            running_label = "Writing placeholder artefacts..."
            completed_label = "Placeholder run completed!"
            failed_label = "Placeholder run failed"
        else:
            running_label = "Running simulation..."
            completed_label = "Simulation run completed!"
            failed_label = "Simulation run failed"

        with st.status(running_label, expanded=True) as status:
            try:
                result = execute_run(
                    resolved_cfg=resolved_cfg,
                    cfg_path=cfg_path_str,
                    mode=exec_mode,
                    progress=progress,
                )
                status.update(label=completed_label, state="complete", expanded=False)
            except Exception as e:
                status.update(label=failed_label, state="error", expanded=True)
                render_failed_state(e)
                return

        if exec_mode == ExecutionMode.REAL:
            render_run_feedback(str(result.run_dir), check_run_outputs(result.run_dir))

        created = to_display_path(result.run_dir)
        set_state(st.session_state, get_state(st.session_state).with_selected_run_dir(created))
        st.session_state["dashboard_last_created_run"] = created
        st.rerun()
