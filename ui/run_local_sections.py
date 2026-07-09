from __future__ import annotations

from typing import Any

import streamlit as st

from ui.run_execution import ExecutionMode
from ui.run_local_catalog import (
    FLOAT_FIELD_SPECS,
    INT_FIELD_SPECS,
    PRESETS,
    BoundedParams,
)


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

    def preset_label(preset_id: str) -> str:
        return options.get(preset_id, preset_id)

    selected_id = st.selectbox(
        "Scenario Preset",
        options=preset_ids,
        index=idx,
        format_func=preset_label,
    )

    for p in PRESETS:
        if p.id == selected_id:
            st.caption(p.description)
            break

    if isinstance(selected_id, str) and selected_id != current_preset_id:
        return selected_id
    return None


def render_bounded_form(current_params: BoundedParams) -> BoundedParams:
    """Render the bounded examiner-facing controls and return updated params."""
    st.subheader("Basic Parameters")
    st.caption(
        "Preset seed behaviour is preserved by default. "
        "Use Advanced Configuration only if you want to override `seeds` explicitly."
    )

    col1, col2 = st.columns(2)

    with col1:
        steps = _render_int_input("steps", current_params.steps)
        rank_alpha = _render_float_input("rank_alpha", current_params.rank_alpha)
        lock_in_threshold = _render_float_input(
            "lock_in_threshold",
            current_params.lock_in_threshold,
        )

    with col2:
        top_k = _render_int_input("top_k", current_params.top_k)
        drift_alpha = _render_float_input("drift_alpha", current_params.drift_alpha)
        persistence_window = _render_int_input(
            "persistence_window",
            current_params.persistence_window,
        )

    return BoundedParams(
        steps=steps,
        top_k=top_k,
        rank_alpha=rank_alpha,
        drift_alpha=drift_alpha,
        lock_in_threshold=lock_in_threshold,
        persistence_window=persistence_window,
    )


def render_advanced_json_section(current_json: str) -> str:
    """Render optional advanced JSON overrides. Returns the JSON string without syncing state."""
    with st.expander("Advanced Configuration (JSON)"):
        st.caption(
            "Optional top-level JSON overrides merged into the selected preset and "
            "basic parameters. only include the fields you want to change."
        )

        st.info(
            "Use this for settings not shown above, such as `seeds`, "
            "`enable_viewpoint_drift`, or interest-update parameters."
        )

        st.markdown("**Common examples**")
        st.markdown("**Example**")
        st.code(
            "{\n"
            '  "seeds": [0, 1, 2],\n'
            '  "enable_viewpoint_drift": false,\n'
            '  "interest_decay": 0.05\n'
            "}",
            language="json",
        )

        advanced_json = st.text_area(
            "JSON Overrides",
            value=current_json,
            height=170,
            placeholder=('{\n  "seeds": [0, 1, 2],\n  "enable_viewpoint_drift": false\n}'),
            help=(
                "Enter a JSON object with only the top-level fields you want to override. "
                'Example: {"seeds": [0, 1, 2]}.'
            ),
        )

        st.caption(
            "Rules: valid JSON only, top-level object only, and preset values remain unchanged "
            "unless overridden here."
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
    st.error(f"**Run Failed**\n\nThe simulation encountered an error:\n`{error}`")


def _render_int_input(field_key: str, current_value: int) -> int:
    spec = INT_FIELD_SPECS[field_key]
    value = st.number_input(
        f"{spec.label} ({spec.bounds_text})",
        min_value=spec.min_value,
        max_value=spec.max_value,
        value=current_value,
        step=spec.step,
        help=spec.help_text,
    )
    return _safe_int(value, current_value)


def _render_float_input(field_key: str, current_value: float) -> float:
    spec = FLOAT_FIELD_SPECS[field_key]
    value = st.number_input(
        f"{spec.label} ({spec.bounds_text})",
        min_value=spec.min_value,
        max_value=spec.max_value,
        value=float(current_value),
        step=spec.step,
        format=spec.format_str,
        help=spec.help_text,
    )
    return _safe_float(value, current_value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
