from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def read_csv_head(path: Path, *, n: int = 200) -> pd.DataFrame:
    # If empty file, return empty DF cleanly.
    if path.exists() and path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path).head(n)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
