from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegrityIssue:
    kind: str
    path: Path
    message: str


def check_run_outputs(run_dir: Path) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []

    if not run_dir.exists():
        return [IntegrityIssue("missing", run_dir, "run_dir does not exist")]
    if not run_dir.is_dir():
        return [IntegrityIssue("invalid", run_dir, "run_dir is not a directory")]

    summary = run_dir / "summary.csv"
    if not summary.exists():
        issues.append(IntegrityIssue("missing", summary, "summary.csv is missing"))
    elif summary.stat().st_size == 0:
        issues.append(IntegrityIssue("empty", summary, "summary.csv is empty"))

    seeds_dir = run_dir / "seeds"
    if not seeds_dir.exists():
        issues.append(IntegrityIssue("missing", seeds_dir, "seeds/ directory is missing"))
        return issues
    if not seeds_dir.is_dir():
        issues.append(IntegrityIssue("invalid", seeds_dir, "seeds exists but is not a directory"))
        return issues

    seed_dirs = sorted([p for p in seeds_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not seed_dirs:
        issues.append(IntegrityIssue("empty", seeds_dir, "No seed subdirectories found in seeds/"))
        return issues

    for sd in seed_dirs:
        run_log = sd / "run_log.csv"
        if not run_log.exists():
            issues.append(IntegrityIssue("missing", run_log, f"{sd.name}: run_log.csv is missing"))
        elif run_log.stat().st_size == 0:
            issues.append(IntegrityIssue("empty", run_log, f"{sd.name}: run_log.csv is empty"))

    return issues
