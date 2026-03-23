from __future__ import annotations

from typing import Any

import streamlit as st

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

    col1, col2 = st.columns(2)
    with col1:
        steps_spec = FIELD_SPECS["steps"]
        steps = st.number_input(
            steps_spec.label,
            min_value=steps_spec.min_value,
            max_value=steps_spec.max_value,
            value=current_params.steps,
            step=steps_spec.step,
            help=steps_spec.help_text,
        )
    with col2:
        top_k_spec = FIELD_SPECS["top_k"]
        top_k = st.number_input(
            top_k_spec.label,
            min_value=top_k_spec.min_value,
            max_value=top_k_spec.max_value,
            value=current_params.top_k,
            step=top_k_spec.step,
            help=top_k_spec.help_text,
        )

        seed_spec = FIELD_SPECS["seed"]
        seed = st.number_input(
            seed_spec.label,
            min_value=seed_spec.min_value,
            max_value=seed_spec.max_value,
            value=current_params.seed,
            step=seed_spec.step,
            help=seed_spec.help_text,
        )

    return BoundedParams(steps=int(steps), top_k=int(top_k), seed=int(seed))


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


def render_run_action_panel(ready_to_run: bool, selected_run_dir: str | None) -> bool:
    st.divider()
    st.subheader("Run")

    st.caption(
        "Placeholder mode writes config_resolved.json + manifest.json only,\n"
        "Real mode runs the simulation and writes real run_log.csv + summary.csv."
    )

    col1, col2 = st.columns([1, 2], vertical_alignment="center")

    with col1:
        run_clicked = st.button("Run heuristic", disabled=not ready_to_run)

    with col2:
        st.write(f"**Selected run dir:** `{selected_run_dir or '(none)'}`")

    return run_clicked


def render_success_banner(last_run: str) -> None:
    """Show the post-run success notice and link to Explore Results."""
    st.success(f"Created run: `{last_run}`")
    col_a, col_b = st.columns([1, 3], vertical_alignment="center")
    with col_a:
        st.page_link("pages/3_Explore_Results.py", label="Open Explore Results", icon="🔍")
    with col_b:
        st.info("Go to **Explore Results** to browse the run directory.")


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
