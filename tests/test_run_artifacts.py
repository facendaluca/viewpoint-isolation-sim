"""
Unit tests for run_artifacts.py utilities.
"""

import re

from fyp_sim.utils.run_artifacts import (
    canonical_json_bytes,
    make_run_id,
    slugify,
    stable_config_hash,
)


def test_slugify():
    assert slugify("Hello World!") == "hello_world"
    assert slugify("Foo-Bar_Baz") == "foo-bar_baz"
    assert slugify("  test  ") == "test"
    assert slugify("invalid/chars") == "invalidchars"


def test_canonical_json():
    obj1 = {"b": 2, "a": 1}
    obj2 = {"a": 1, "b": 2}
    assert canonical_json_bytes(obj1) == canonical_json_bytes(obj2)
    assert canonical_json_bytes(obj1) == b'{"a":1,"b":2}'


def test_stable_config_hash():
    cfg1 = {"seeds": [1, 2], "alpha": 0.5}
    cfg2 = {"alpha": 0.5, "seeds": [1, 2]}
    cfg3 = {"alpha": 0.6, "seeds": [1, 2]}

    hash1 = stable_config_hash(cfg1)
    hash2 = stable_config_hash(cfg2)
    hash3 = stable_config_hash(cfg3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA256 hex


def test_make_run_id():
    run_id = make_run_id(
        exp_name="test_exp",
        config_slug="base_config",
        corpus_mode="file",
        base_seed=42,
        n_runs=10,
        hash8="abcdef12",
    )

    # Expected format: <timestamp>__test_exp__base_config__file__seed42__n10__abcdef12
    pattern = r"^\d{8}-\d{6}__test_exp__base_config__file__seed42__n10__abcdef12$"
    assert re.match(pattern, run_id)
