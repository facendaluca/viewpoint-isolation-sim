from __future__ import annotations

import pytest

from ui.run_local_form import (
    BoundedParams,
    build_final_overrides,
    get_preset_by_name,
    parse_advanced_json,
)


def test_bounded_params_defaults():
    params = BoundedParams.from_dict({})
    assert params.steps == 150
    assert params.top_k == 5
    assert params.seed == 0


def test_bounded_params_custom_dict():
    params = BoundedParams.from_dict({"steps": 10, "top_k": 2, "seed": 42, "ignored": "yes"})
    assert params.steps == 10
    assert params.top_k == 2
    assert params.seed == 42


def test_get_preset_by_name():
    preset = get_preset_by_name("Quick Test")
    assert preset is not None
    assert preset.name == "Quick Test"
    assert preset.params.steps == 50

    none_preset = get_preset_by_name("Nonexistent")
    assert none_preset is None


def test_parse_advanced_json_success():
    res = parse_advanced_json('{"custom_alpha": 0.5}')
    assert res == {"custom_alpha": 0.5}


def test_parse_advanced_json_empty():
    assert parse_advanced_json("") == {}
    assert parse_advanced_json("   ") == {}


def test_parse_advanced_json_not_dict():
    with pytest.raises(ValueError, match="JSON object"):
        parse_advanced_json("[1, 2, 3]")


def test_parse_advanced_json_invalid():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_advanced_json("{bad json")


def test_build_final_overrides():
    params = BoundedParams(steps=100, top_k=2, seed=99)
    # Advanced JSON overrides basic param "steps" and adds a new one "new_feature"
    advanced = '{"steps": 200, "new_feature": true}'

    final = build_final_overrides(params, advanced)
    assert final["steps"] == 200
    assert final["top_k"] == 2
    assert final["seed"] == 99
    assert final["new_feature"] is True
