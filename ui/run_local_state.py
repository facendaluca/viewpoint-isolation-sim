from __future__ import annotations

import streamlit as st

from ui.app_state import get_state, set_state
from ui.run_local_form import BoundedParams, build_final_overrides, get_preset_by_name

_ADVANCED_JSON_KEY = "run_local_advanced_json"
_SELECTED_PRESET_KEY = "run_local_selected_preset"


def init_page_state() -> None:
    """Initialize local widget state from global AppState if not yet set."""
    # Ensure keys exist so we don't get KeyError on first render
    if _ADVANCED_JSON_KEY not in st.session_state:
        st.session_state[_ADVANCED_JSON_KEY] = "{}"

    if _SELECTED_PRESET_KEY not in st.session_state:
        st.session_state[_SELECTED_PRESET_KEY] = "Custom"


def apply_preset(preset_name: str) -> None:
    preset = get_preset_by_name(preset_name)
    if preset:
        app_state = get_state(st.session_state)
        new_state = app_state.with_selected_scenario(preset.scenario).with_params(
            preset.params.to_dict()
        )
        set_state(st.session_state, new_state)

        # Clear advanced JSON when applying a preset for a clean start
        st.session_state[_ADVANCED_JSON_KEY] = "{}"
        st.session_state[_SELECTED_PRESET_KEY] = preset_name
    elif preset_name == "Custom":
        st.session_state[_SELECTED_PRESET_KEY] = "Custom"


def sync_form_to_app_state(
    scenario: str, form_params: BoundedParams, advanced_json: str
) -> tuple[bool, str]:
    """
    Attempt to sync form values back to global AppState.
    Returns (success, error_message)
    """
    try:
        final_overrides = build_final_overrides(form_params, advanced_json)

        app_state = get_state(st.session_state)
        new_state = app_state.with_selected_scenario(scenario).with_params(final_overrides)
        set_state(st.session_state, new_state)

        st.session_state[_ADVANCED_JSON_KEY] = advanced_json

        return True, ""
    except ValueError as e:
        return False, str(e)


def get_current_form_params() -> BoundedParams:
    """Extract bounded params from current AppState."""
    app_state = get_state(st.session_state)
    return BoundedParams.from_dict(app_state.params)
