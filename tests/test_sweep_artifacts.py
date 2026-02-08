"""
Integration tests for sweep runner and plotting artifacts.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_plots import main as plot_main
from scripts.run_sweep import main as sweep_main


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_plotting_pipeline(temp_output_dir):
    """Test standard plotting pipeline with mock data."""
    # Setup mock sweep summary
    sweep_csv = temp_output_dir / "sweep_summary.csv"
    data = [
        {"top_k": 5, "alpha": 0.0, "mean_vii_mean": 0.1, "lock_in_rate_mean": 0.05},
        {"top_k": 5, "alpha": 0.5, "mean_vii_mean": 0.2, "lock_in_rate_mean": 0.10},
        {"top_k": 10, "alpha": 0.0, "mean_vii_mean": 0.15, "lock_in_rate_mean": 0.02},
        {"top_k": 10, "alpha": 0.5, "mean_vii_mean": 0.25, "lock_in_rate_mean": 0.08},
    ]
    df = pd.DataFrame(data)
    df.to_csv(sweep_csv, index=False)

    out_plots = temp_output_dir / "plots"

    # Call make_plots via CLI args simulation
    import sys

    original_argv = sys.argv
    sys.argv = [
        "make_plots.py",
        "--sweep-summary",
        str(sweep_csv),
        "--out-dir",
        str(out_plots),
    ]

    try:
        assert plot_main() == 0
    finally:
        sys.argv = original_argv

    assert (out_plots / "heatmap_mean_vii.png").exists()
    assert (out_plots / "heatmap_lock_in_rate.png").exists()


def test_sweep_dry_run_manifest(monkeypatch, temp_output_dir):
    """Test sweep dry-run generates valid manifest plan."""

    # Mock config loading to avoid file dependency
    def mock_load(path):
        return {
            "steps": 10,
            "seeds": [42],
            "top_k_grid": [5],
            "alpha_grid": [0.0],
            "lock_in_threshold": 0.8,
            "persistence_window": 5,
            "user": {
                "phenotype": "watcher",
                "viewpoint_score": 0.5,
                "interest_vector": {"topic": 1.0},
                "sentiment_threshold": -0.5,
            },
            "video_pool": "outputs/test_pool.json",  # Dummy
        }

    monkeypatch.setattr("scripts.run_sweep.load_config", mock_load)

    # Mock exists check
    monkeypatch.setattr("pathlib.Path.exists", lambda s: True)

    import sys
    from io import StringIO

    captured_output = StringIO()
    original_argv = sys.argv
    sys.argv = [
        "run_sweep.py",
        "dummy_config.json",
        "--dry-run",
        "--corpus-mode",
        "generated",
        "--corpus-size",
        "100",
        "--corpus-seed",
        "123",
    ]

    sys.stdout = captured_output

    try:
        assert sweep_main() == 0
    finally:
        sys.argv = original_argv
        sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "Dry run complete" in output
    assert '"corpus": {' in output
    assert '"source": "generated"' in output
