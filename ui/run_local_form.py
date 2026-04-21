"""
Form processing for the Run Locally page.

Owns: raw-to-BoundedParams coercion, advanced JSON extraction/parsing,
merged override validation, and final submission building.
No Streamlit. No catalogue data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ui.run_local_catalog import (
    BOUND_PARAM_KEYS,
    FLOAT_FIELD_SPECS,
    INT_FIELD_SPECS,
    BoundedParams,
    FloatFieldSpec,
    IntFieldSpec,
    Preset,
    get_preset,
)


@dataclass(frozen=True)
class RunLocalFormInput:
    preset_id: str
    params: BoundedParams
    advanced_json: str = ""


@dataclass(frozen=True)
class RunLocalBuildResult:
    preset: Preset
    overrides: dict[str, Any]
    errors: tuple[str, ...]


def params_from_raw(
    raw_params: dict[str, Any] | None,
    *,
    defaults: BoundedParams | None = None,
) -> BoundedParams:
    """Reconstruct safe BoundedParams from an arbitrary raw params dict."""
    raw = raw_params if isinstance(raw_params, dict) else {}
    fallback = defaults or get_preset("E1_baseline_single_watcher").defaults

    return BoundedParams(
        steps=_coerce_int(raw.get("steps"), spec=INT_FIELD_SPECS["steps"], default=fallback.steps),
        top_k=_coerce_int(raw.get("top_k"), spec=INT_FIELD_SPECS["top_k"], default=fallback.top_k),
        rank_alpha=_coerce_float(
            raw.get("rank_alpha"),
            spec=FLOAT_FIELD_SPECS["rank_alpha"],
            default=fallback.rank_alpha,
        ),
        drift_alpha=_coerce_float(
            raw.get("drift_alpha"),
            spec=FLOAT_FIELD_SPECS["drift_alpha"],
            default=fallback.drift_alpha,
        ),
        lock_in_threshold=_coerce_float(
            raw.get("lock_in_threshold"),
            spec=FLOAT_FIELD_SPECS["lock_in_threshold"],
            default=fallback.lock_in_threshold,
        ),
        persistence_window=_coerce_int(
            raw.get("persistence_window"),
            spec=INT_FIELD_SPECS["persistence_window"],
            default=fallback.persistence_window,
        ),
    )


def extract_advanced_json(raw_params: dict[str, Any] | None) -> str:
    """Produce the advanced-JSON text for non-bounded keys already stored in params."""
    raw = raw_params if isinstance(raw_params, dict) else {}
    advanced = {k: v for k, v in raw.items() if k not in BOUND_PARAM_KEYS}
    return json.dumps(advanced, indent=2, sort_keys=True) if advanced else ""


def parse_advanced_json(text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse and sanity-check the raw advanced JSON text. Returns (parsed, errors)."""
    raw = text.strip()
    if not raw:
        return {}, []

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as err:
        return {}, [f"Advanced JSON is invalid: {err.msg} (line {err.lineno}, col {err.colno})."]

    if not isinstance(loaded, dict):
        return {}, ["Advanced JSON must be a JSON object at the top level."]

    return loaded, []


def validate_overrides(overrides: dict[str, Any]) -> list[str]:
    """Validate merged overrides against field specs and seeds rules."""
    errors: list[str] = []

    for key, spec in INT_FIELD_SPECS.items():
        if key not in overrides:
            continue
        value = overrides[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"`{key}` must be an integer between {spec.bounds_text}.")
            continue
        if value < spec.min_value or value > spec.max_value:
            errors.append(f"`{key}` must be between {spec.bounds_text}.")

    for key, spec in FLOAT_FIELD_SPECS.items():
        if key not in overrides:
            continue
        value = overrides[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"`{key}` must be a number between {spec.bounds_text}.")
            continue
        numeric_value = float(value)
        if numeric_value < spec.min_value or numeric_value > spec.max_value:
            errors.append(f"`{key}` must be between {spec.bounds_text}.")

    if "seeds" in overrides:
        errors.extend(_validate_seeds(overrides["seeds"]))

    return errors


def build_submission(form_input: RunLocalFormInput) -> RunLocalBuildResult:
    """Merge widget params + advanced JSON and validate the combined overrides."""
    preset = get_preset(form_input.preset_id)
    advanced_overrides, parse_errors = parse_advanced_json(form_input.advanced_json)

    overrides = dict(form_input.params.to_dict())
    overrides.update(advanced_overrides)

    return RunLocalBuildResult(
        preset=preset,
        overrides=overrides,
        errors=tuple([*parse_errors, *validate_overrides(overrides)]),
    )


def _validate_seeds(seeds_raw: Any) -> list[str]:
    seed_spec = IntFieldSpec(
        key="seed",
        label="Random seed",
        min_value=0,
        max_value=99999,
        default=0,
        step=1,
        help_text="Deterministic seed used for the run.",
    )

    if not isinstance(seeds_raw, list) or not seeds_raw:
        return ["`seeds` must be a non-empty list of integers."]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in seeds_raw):
        return ["`seeds` must contain integers only."]

    errors: list[str] = []
    if any(x < seed_spec.min_value or x > seed_spec.max_value for x in seeds_raw):
        errors.append(f"`seeds` values must be between {seed_spec.bounds_text}.")
    if len(seeds_raw) > 20:
        errors.append("`seeds` must contain at most 20 values.")
    return errors


def _coerce_int(raw: Any, *, spec: IntFieldSpec, default: int) -> int:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        candidate = raw
    elif isinstance(raw, float) and raw.is_integer():
        candidate = int(raw)
    else:
        return default
    return max(spec.min_value, min(spec.max_value, candidate))


def _coerce_float(raw: Any, *, spec: FloatFieldSpec, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int | float):
        candidate = float(raw)
    else:
        return default
    return max(spec.min_value, min(spec.max_value, candidate))
