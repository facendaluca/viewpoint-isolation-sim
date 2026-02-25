from __future__ import annotations

import sys
from collections.abc import Callable

from fyp_sim.models import ConfigValidationError


def run_cli(main: Callable[[], None]) -> None:
    """Run a script() with clean, readable config-failure output."""
    try:
        main()
    except ConfigValidationError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(2) from None
