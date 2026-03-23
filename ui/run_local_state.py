from __future__ import annotations

import streamlit as st

from ui.app_state import get_state, set_state
from ui.run_local_catalog import DEFAULT_PRESET_ID, get_preset, infer_preset_id
from ui.run_local_form import extract_advanced_json

_PRESET_ID_KEY = "run_local_preset_id"
_ADVANCED_JSON_KEY = "run_local_advanced_json"


def init_page_state() -> None:
    """Hydrate widget session keys from the global AppState upon first load."""
    if _PRESET_ID_KEY in st.session_state and _ADVANCED_JSON_KEY in st.session_state:
        return

    app_state = get_state(st.session_state)

    preset_id = infer_preset_id(app_state.selected_scenario)
    st.session_state[_PRESET_ID_KEY] = preset_id

    st.session_state[_ADVANCED_JSON_KEY] = extract_advanced_json(app_state.params)


def handle_preset_change(new_preset_id: str) -> None:
    preset = get_preset(new_preset_id)
    app_state = get_state(st.session_state)

    # Update app state immediately to reflect scenario and default params
    new_state = app_state.with_selected_scenario(preset.scenario).with_params(
        preset.defaults.to_dict()
    )
    set_state(st.session_state, new_state)

    # Set widget session state keys directly so they survive the rerun
    st.session_state[_PRESET_ID_KEY] = preset.id
    st.session_state[_ADVANCED_JSON_KEY] = ""


def get_current_preset_id() -> str:
    return st.session_state.get(_PRESET_ID_KEY, DEFAULT_PRESET_ID)


def get_current_advanced_json() -> str:
    return st.session_state.get(_ADVANCED_JSON_KEY, "")


def sync_valid_build(scenario: str, final_overrides: dict, advanced_json: str) -> None:
    """Sync a validated run submission back to the global AppState and session storage."""
    app_state = get_state(st.session_state)
    new_state = app_state.with_selected_scenario(scenario).with_params(final_overrides)
    set_state(st.session_state, new_state)

    st.session_state[_ADVANCED_JSON_KEY] = advanced_json
