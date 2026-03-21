from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ui.run_inspector import find_config_file, find_manifest_file, find_summary_file


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    config_path: Path | None
    manifest_path: Path | None
    summary_path: Path | None
    plots_dir: Path
    seeds_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> RunContext:
        return cls(
            run_dir=run_dir,
            config_path=find_config_file(run_dir),
            manifest_path=find_manifest_file(run_dir),
            summary_path=find_summary_file(run_dir),
            plots_dir=run_dir / "plots",
            seeds_dir=run_dir / "seeds",
        )
