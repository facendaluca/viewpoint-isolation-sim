from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STATE_KEY = "examiner_dashboard_state_v1"

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/


def resolve_repo_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def to_display_path(p: str | Path) -> str:
    path = Path(p)
    if not path.is_absolute():
        path = resolve_repo_path(path)
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class AppState:
    """
    UI-only state for the dashboard. Keep this small and stable.

    Stored in Streamlit session_state as a plain dict to avoid coupling tests to Streamlit.
    """

    selected_run_dir: str | None = None
    selected_scenario: str = "experiment_baseline"
    params: dict[str, Any] = field(default_factory=dict)
    selected_run_a: str | None = None
    selected_run_b: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_run_dir": self.selected_run_dir,
            "selected_scenario": self.selected_scenario,
            "params": dict(self.params),
            "selected_run_a": self.selected_run_a,
            "selected_run_b": self.selected_run_b,
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

        selected_run_a = d.get("selected_run_a")
        if selected_run_a is not None and not isinstance(selected_run_a, str):
            selected_run_a = str(selected_run_a)

        selected_run_b = d.get("selected_run_b")
        if selected_run_b is not None and not isinstance(selected_run_b, str):
            selected_run_b = str(selected_run_b)

        return AppState(
            selected_run_dir=selected_run_dir,
            selected_scenario=selected_scenario,
            params=params,
            selected_run_a=selected_run_a,
            selected_run_b=selected_run_b,
        )

    def with_selected_run_dir(self, run_dir: str | None) -> AppState:
        return AppState(
            selected_run_dir=run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(self.params),  # type: ignore[arg-type]
            selected_run_a=self.selected_run_a,
            selected_run_b=self.selected_run_b,
        )

    def with_selected_scenario(self, scenario: str) -> AppState:
        scenario = scenario.strip() or "experiment_baseline"
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=scenario,
            params=dict(self.params),
            selected_run_a=self.selected_run_a,
            selected_run_b=self.selected_run_b,
        )

    def with_params(self, params: dict[str, Any]) -> AppState:
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(params),
            selected_run_a=self.selected_run_a,
            selected_run_b=self.selected_run_b,
        )

    def with_selected_run_a(self, run_dir: str | None) -> AppState:
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(self.params),
            selected_run_a=run_dir,
            selected_run_b=self.selected_run_b,
        )

    def with_selected_run_b(self, run_dir: str | None) -> AppState:
        return AppState(
            selected_run_dir=self.selected_run_dir,
            selected_scenario=self.selected_scenario,
            params=dict(self.params),
            selected_run_a=self.selected_run_a,
            selected_run_b=run_dir,
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
    """
    List run directories.

    Supports both layouts:
    1) New: outputs/runs/YYYYMMDD/<run_id>
    2) Old: outputs/runs/run_id/
    """
    base = resolve_repo_path(base_dir)
    if not base.exists() or not base.is_dir():
        return []

    def is_date_dir(p: Path) -> bool:
        return p.name.isdigit() and len(p.name) == 8

    def looks_like_run_dir(p: Path) -> bool:
        markers = (
            "manifest.json",
            "config_resolved.json",
            "summary.csv",
            "config.json",
            "meta.json",
        )
        return any((p / m).exists() for m in markers)

    run_dirs: list[Path] = []

    # If base contains run dirs directly (old layout), include them.
    for p in sorted(base.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if is_date_dir(p):
            for r in sorted([d for d in p.iterdir() if d.is_dir()], key=lambda x: x.name):
                run_dirs.append(r)
            continue
        if looks_like_run_dir(p):
            run_dirs.append(p)

    return [to_display_path(p) for p in run_dirs]


def available_scenarios(config_dir: str | Path = "configs") -> list[str]:
    """
    List scenario names based on config JSON files.
    We return stems (e.g experiment_baseline) to keep it simple and stable.
    """
    cfg = resolve_repo_path(config_dir)
    if not cfg.exists() or not cfg.is_dir():
        return ["experiment_baseline"]

    candidates = sorted(cfg.glob("*json"), key=lambda p: p.name)
    stems = [p.stem for p in candidates]
    # Always include a sane default even if configs dir is empty
    return stems or ["experiment_baseline"]
