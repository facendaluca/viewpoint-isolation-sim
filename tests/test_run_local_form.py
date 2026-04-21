from ui.run_local_catalog import BoundedParams, get_preset, infer_preset_id
from ui.run_local_form import (
    RunLocalFormInput,
    build_submission,
    extract_advanced_json,
    params_from_raw,
    parse_advanced_json,
    validate_overrides,
)


def test_get_preset_returns_fallback_for_unknown():
    preset = get_preset("invalid_id_xyz")
    assert preset.id == "E1_baseline_single_watcher"


def test_infer_preset_id():
    assert infer_preset_id("E1_baseline_single_watcher") == "E1_baseline_single_watcher"
    assert infer_preset_id("E2_baseline_multi_phenotype_cohort") == "E2_baseline_multi_phenotype_cohort"
    assert infer_preset_id("unknown_scenario") == "E1_baseline_single_watcher"


def test_params_from_raw_handles_empty():
    params = params_from_raw(None)
    assert params.steps == 200
    assert params.top_k == 5
    assert params.rank_alpha == 0.30


def test_params_from_raw_handles_partial_dict():
    params = params_from_raw({"steps": 300})
    assert params.steps == 300
    assert params.top_k == 5
    assert params.rank_alpha == 0.30


def test_params_from_raw_ignores_seeds_list():
    # seeds are now handled via advanced JSON, not BoundedParams
    params = params_from_raw({"seeds": [42, 43]})
    assert params.steps == 200


def test_extract_advanced_json():
    raw = {"steps": 100, "custom_alpha": 0.5, "top_k": 3}
    ext = extract_advanced_json(raw)
    assert "custom_alpha" in ext
    assert "steps" not in ext
    assert "top_k" not in ext


def test_parse_advanced_json_success():
    res, errs = parse_advanced_json('{"custom_alpha": 0.5}')
    assert res == {"custom_alpha": 0.5}
    assert not errs


def test_parse_advanced_json_invalid():
    res, errs = parse_advanced_json("{bad json")
    assert res == {}
    assert len(errs) == 1
    assert "Advanced JSON is invalid" in errs[0]


def test_parse_advanced_json_not_dict():
    res, errs = parse_advanced_json("[1, 2, 3]")
    assert res == {}
    assert len(errs) == 1
    assert "JSON object at the top level" in errs[0]


def test_validate_overrides_bounds():
    errs = validate_overrides({"steps": 9999, "top_k": 5})
    assert len(errs) == 1
    assert "between 25–500" in errs[0]


def test_validate_overrides_types():
    errs = validate_overrides({"steps": "100"})
    assert len(errs) == 1
    assert "must be an integer" in errs[0]


def test_validate_overrides_seeds_list():
    errs = validate_overrides({"seeds": [42, 999999]})
    assert len(errs) == 1
    assert "values must be between" in errs[0]

    errs_invalid = validate_overrides({"seeds": "42"})
    assert len(errs_invalid) == 1
    assert "must be a non-empty list of integers" in errs_invalid[0]


def _make_params(steps: int = 100, top_k: int = 2) -> BoundedParams:
    return BoundedParams(
        steps=steps,
        top_k=top_k,
        rank_alpha=0.30,
        drift_alpha=0.02,
        lock_in_threshold=0.20,
        persistence_window=10,
    )


def test_build_submission_clean():
    form_input = RunLocalFormInput(
        preset_id="E1_baseline_single_watcher",
        params=_make_params(),
        advanced_json='{"extra_flag": true}',
    )
    result = build_submission(form_input)
    assert not result.errors
    assert result.overrides["steps"] == 100
    assert result.overrides["extra_flag"] is True
    assert result.preset.id == "E1_baseline_single_watcher"


def test_build_submission_with_errors():
    form_input = RunLocalFormInput(
        preset_id="E1_baseline_single_watcher",
        params=_make_params(),
        advanced_json='{"steps": 9999}',  # Out of bounds
    )
    result = build_submission(form_input)
    assert result.errors
    assert "between 25–500" in result.errors[0]
