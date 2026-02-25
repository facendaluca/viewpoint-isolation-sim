from __future__ import annotations

import pytest

from fyp_sim.cli import run_cli
from fyp_sim.models import ConfigValidationError


def test_run_cli_prints_clean_config_error_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    def bad_main() -> None:
        raise ConfigValidationError("bad config")

    with pytest.raises(SystemExit) as e:
        run_cli(bad_main)

    assert e.value.code == 2
    out = capsys.readouterr()
    assert out.err.strip() == "Config error: bad config"
