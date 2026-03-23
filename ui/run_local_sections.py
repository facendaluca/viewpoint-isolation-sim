from __future__ import annotations

from typing import Any

import streamlit as st

from ui.run_execution import ExecutionMode
from ui.run_local_catalog import FIELD_SPECS, PRESETS, BoundedParams


def render_preset_selector(current_preset_id: str) -> str | None:
    """
    Render preset selector.
    Returns the selected preset_id if it differs from current, or None.
    Does NOT mutate state or call rerun.
    """
    st.subheader("Presets")

    options = {p.id: p.label for p in PRESETS}
    preset_ids = list(options.keys())

    try:
        idx = preset_ids.index(current_preset_id)
    except ValueError:
        idx = 0

    st.write("Start from a preset scenario or configure advanced options below.")

    selected_id = st.selectbox(
        "Scenario Preset",
        options=preset_ids,
        index=idx,
        format_func=lambda x: options.get(x, x),
    )

    for p in PRESETS:
        if p.id == selected_id:
            st.caption(p.description)
            break

    if isinstance(selected_id, str) and selected_id != current_preset_id:
        return selected_id
    return None


def render_bounded_form(current_params: BoundedParams) -> BoundedParams:
    """Render basic widget inputs without any scenario free-text lookup. Return updated BoundedParams."""
    st.subheader("Basic Parameters")

    def _safe_int(val: Any, default: int) -> int:
        try:
            return int(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    col1, col2 = st.columns(2)
    with col1:
        steps_spec = FIELD_SPECS["steps"]
        steps = st.number_input(
            f"{steps_spec.label} ({steps_spec.bounds_text})",
            value=current_params.steps,
            step=steps_spec.step,
            help=steps_spec.help_text,
        )
    with col2:
        top_k_spec = FIELD_SPECS["top_k"]
        top_k = st.number_input(
            f"{top_k_spec.label} ({top_k_spec.bounds_text})",
            value=current_params.top_k,
            step=top_k_spec.step,
            help=top_k_spec.help_text,
        )

        seed_spec = FIELD_SPECS["seed"]
        seed = st.number_input(
            f"{seed_spec.label} ({seed_spec.bounds_text})",
            value=current_params.seed,
            step=seed_spec.step,
            help=seed_spec.help_text,
        )

    return BoundedParams(
        steps=_safe_int(steps, current_params.steps),
        top_k=_safe_int(top_k, current_params.top_k),
        seed=_safe_int(seed, current_params.seed),
    )


def render_advanced_json_section(current_json: str) -> str:
    """Render optional advanced JSON overrides. Returns the JSON string without syncing state."""
    with st.expander("Advanced Configuration (JSON)"):
        st.caption(
            "These overrides merge with the parameters above. Invalid JSON will block the run."
        )

        advanced_json = st.text_area(
            "JSON Overrides",
            value=current_json,
            height=150,
        )
        return str(advanced_json)


def render_validation_errors(errors: tuple[str, ...]) -> None:
    """Presents building or validation errors directly without blocking the UI cycle early."""
    if errors:
        for err in errors:
            st.error(err, icon="⚠️")


def render_resolved_config_preview(
    ready_to_preview: bool,
    resolved_cfg: dict | None,
    cfg_path_str: str | None,
) -> None:
    with st.expander("Resolved config preview", expanded=False):
        if not ready_to_preview:
            st.info("Fix configuration errors to preview the resolved config.")
        elif not resolved_cfg:
            st.info("Resolved config unavailable (check scenario file/path).")
        else:
            st.caption(f"Loaded from: `{cfg_path_str}`")
            st.json(resolved_cfg)


def render_run_action_panel(
    ready_to_run: bool, selected_run_dir: str | None, exec_mode: ExecutionMode
) -> bool:
    st.divider()
    st.subheader("Run")

    if exec_mode == ExecutionMode.PLACEHOLDER:
        st.caption("Placeholder mode writes lightweight artefacts only.")
    else:
        st.caption("Real mode runs the simulation and produces real outputs.")

    if ready_to_run:
        st.success("Configuration validated and resolved. Ready to run.", icon="✅")
    else:
        st.info("Please resolve configuration issues above before running.", icon="ℹ️")

    col1, col2 = st.columns([1, 2], vertical_alignment="center")

    with col1:
        run_clicked = st.button("Run heuristic", disabled=not ready_to_run)

    with col2:
        st.write(f"**Selected run dir:** `{selected_run_dir or '(none)'}`")

    return run_clicked


def render_success_banner(last_run: str, run_mode: str | None = None) -> None:
    """Show the post-run success notice and link to Explore Results."""
    if run_mode == "placeholder":
        summary = (
            "Placeholder run finished successfully. Outputs were created, "
            "and this run is now selected for inspection."
        )
    elif run_mode == "real":
        summary = (
            "Real simulation run finished successfully. Outputs were created, "
            "and this run is now selected for inspection."
        )
    else:
        summary = "Run finished successfully and is now selected for inspection."

    st.success(f"**Run completed successfully.** Created run: `{last_run}`", icon="🎉")

    col_a, col_b = st.columns([2, 1], vertical_alignment="top")
    with col_a:
        st.markdown("**Created run directory**")
        st.code(last_run)
        st.caption(summary)
        st.caption("This run is now the selected run directory for the current session.")

    with col_b:
        st.markdown("**Next step**")
        st.page_link("pages/3_Explore_Results.py", label="Open Explore Results", icon="🔍")
        st.info("Use Explore Results to inspect the outputs of this run.")


def render_execution_mode() -> str:
    """Render execution mode selector. Returns the selected label string."""
    mode_label = st.radio(
        "Execution mode",
        options=["Placeholder (fast)", "Real simulation (writes real logs)"],
        index=0,
        horizontal=True,
    )
    if mode_label == "Real simulation (writes real logs)":
        st.info(
            "Real simulation runs in **heuristic mode only** in the hosted Examiner Dashboard. "
            "LLM mode is supported only when running the repo locally (see README.md)."
        )
    return str(mode_label)


def render_run_feedback(
    run_dir: str,
    issues: list[Any],
) -> None:
    """Display integrity issues (if any) after a real simulation run."""
    if not issues:
        return
    st.warning("Run completed, but some outputs look incomplete:")
    for iss in issues[:8]:
        st.write(f"- **{iss.kind}**: `{iss.path}` - {iss.message}")
    if len(issues) > 8:
        st.caption(f"Showing 8 of {len(issues)} issues.")


def render_failed_state(error: Exception) -> None:
    """Show a clear error state when execution fails."""
    st.error(f"**Run Failed**\n\nThe simulation encountered an error:\n`{error}`", icon="🚨")
