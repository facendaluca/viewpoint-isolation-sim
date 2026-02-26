from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STATE_KEY = "examiner_dashboard_state_v1"


@dataclass(frozen=True)
class AppState:
    """
    UI-only state for the dashboard. Keep this small and stable.

    Stored in Streamlit session_state as a plain dict to avoid coupling tests to Streamlit.
    """

    selected_run_dir: str | None = None
    selected_scenario: str = "experiment_baseline"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_run_dir": self.selected_run_dir,
            "selected_scenario": self.selected_scenario,
            "params": dict(self.params),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> AppState:
        selected_run_dir = d.get("selected_run_dir")
        if selected_run_dir is not None and not isinstance(selected_run_dir, str):
            selected_run_dir = str(selected_run_dir)

        selected_scenario = d.get("selected_scenario", "experiment_baseline")
        if not isinstance(selected_scenario, str) or not selected_scenario:
            selected_scenario = "experiment_baseline"

        params_raw = d.get("params", {})
        params = dict(params_raw) if isinstance(params_raw, dict) else {}

        return AppState(
            selected_run_dir=selected_run_dir,
            selected_scenario=selected_scenario,
            params=params,
        )

    def with_selected_run_dir(self, run_dir: str | None) -> AppState:
        return AppState(
            selected_run_dir=run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(self.params),  # type: ignore[arg-type]
        )

    def with_selected_scenario(self, scenario: str) -> AppState:
        scenario = scenario.strip() or "experiment_baseline"
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=scenario,
            params=dict(self.params),
        )

    def with_params(self, params: dict[str, Any]) -> AppState:
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(params),
        )


def get_state(session: MutableMapping[str, Any]) -> AppState:
    """Get state from a session mapping (e.g. st.session_state), initialising if missing."""
    raw = session.get(_STATE_KEY)
    if isinstance(raw, dict):
        return AppState.from_dict(raw)

    # Initialise with default
    state = AppState()
    session[_STATE_KEY] = state.to_dict()
    return state


def set_state(session: MutableMapping[str, Any], state: AppState) -> None:
    session[_STATE_KEY] = state.to_dict()


def available_runs(base_dir: str | Path = "outputs/runs") -> list[str]:
    """List run directories (UI-only filesystem browse; no simulation logic)."""
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    # Stable ordering for reproducibility
    return [str(p) for p in sorted(runs, key=lambda x: x.name)]


def available_scenarios(config_dir: str | Path = "configs") -> list[str]:
    """
    List scenario names based on config JSON files.
    We return stems (e.g experiment_baseline) to keep it simple and stable.
    """
    cfg = Path(config_dir)
    if not cfg.exists() or not cfg.is_dir():
        return ["experiment_baseline.json"]

    candidates = sorted(cfg.glob("*json"), key=lambda p: p.name)
    stems = [p.stem for p in candidates]
    # Always include a sane default even if configs dir is empty
    return stems or ["experiment_baseline"]
