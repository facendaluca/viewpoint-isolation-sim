from __future__ import annotations

from typing import Any

import streamlit as st

from ui.run_local_form import PRESETS, BoundedParams
from ui.run_local_state import (
    _ADVANCED_JSON_KEY,
    _SELECTED_PRESET_KEY,
    apply_preset,
    get_current_form_params,
)


def render_preset_selector() -> None:
    st.subheader("Presets")
    preset_names = ["Custom"] + [p.name for p in PRESETS]

    current_preset = st.session_state.get(_SELECTED_PRESET_KEY, "Custom")
    idx = preset_names.index(current_preset) if current_preset in preset_names else 0

    st.write("Start from a preset scenario or configure custom options below.")

    selected = st.selectbox("Scenario Preset", options=preset_names, index=idx)
    if selected != current_preset:
        apply_preset(selected)
        st.rerun()


def render_bounded_form(current_scenario: str) -> tuple[str, BoundedParams]:
    """Render basic widget inputs. Return updated scenario string and updated BoundedParams."""
    st.subheader("Basic Parameters")

    current_params = get_current_form_params()

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.text_input("Scenario", value=current_scenario)
        steps = st.number_input("Steps", min_value=1, max_value=1000, value=current_params.steps)
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=current_params.top_k)
        seed = st.number_input("Seed", min_value=0, max_value=99999, value=current_params.seed)

    new_params = BoundedParams(steps=int(steps), top_k=int(top_k), seed=int(seed))
    return scenario, new_params


def render_advanced_json_section() -> str:
    """Render optional advanced JSON overrides. Returns the JSON string."""
    with st.expander("Advanced Configuration (JSON)"):
        st.caption(
            "These overrides merge with the parameters above. Invalid JSON will block the run."
        )
        current_json = st.session_state.get(_ADVANCED_JSON_KEY, "{}")

        advanced_json = st.text_area(
            "JSON Overrides",
            value=current_json,
            height=150,
        )
        return advanced_json


def render_resolved_config_preview(
    parsed_ok: bool,
    resolved_ok: bool,
    resolved_cfg: dict[str, Any] | None,
    cfg_path_str: str | None,
) -> None:
    with st.expander("Resolved config preview", expanded=False):
        if not parsed_ok:
            st.info("Fix Advanced JSON to preview the resolved config.")
        elif not resolved_ok:
            st.info("Resolved config unavailable (check scenario file/path).")
        else:
            st.caption(f"Loaded from: `{cfg_path_str}`")
            st.json(resolved_cfg)


def render_run_action_panel(ready_to_run: bool, selected_run_dir: str | None) -> bool:
    """
    Renders the action buttons and the selected run directory overview.
    Returns True if the run button was clicked.
    """
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
