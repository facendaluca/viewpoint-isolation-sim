from __future__ import annotations

from pathlib import Path


def _first_existing_file(run_dir: Path, names: list[str]) -> Path | None:
    for n in names:
        p = run_dir / n
        if p.is_file():
            return p
    return None


def find_config_file(run_dir: Path) -> Path | None:
    """Find the best config file (resolved preferred over raw)."""
    return _first_existing_file(run_dir, ["config_resolved.json", "config.json"])


def find_manifest_file(run_dir: Path) -> Path | None:
    """Find the manifest file (manifest preferred over meta)."""
    return _first_existing_file(run_dir, ["manifest.json", "meta.json"])


def find_summary_file(run_dir: Path) -> Path | None:
    """Find the summary file."""
    return _first_existing_file(run_dir, ["summary.csv"])


def list_seed_dirs(run_dir: Path) -> list[Path]:
    """List all valid seed directories in the run (e.g., seeds/s00042)."""
    seeds_dir = run_dir / "seeds"
    if not seeds_dir.is_dir():
        return []

    # Return all subdirectories sorted by name
    return sorted((p for p in seeds_dir.iterdir() if p.is_dir()), key=lambda x: x.name)


def find_seed_log(seed_dir: Path) -> Path | None:
    """Find the run log file for a specific seed."""
    log_path = seed_dir / "run_log.csv"
    if log_path.is_file():
        return log_path
    return None


def list_files(run_dir: Path) -> list[str]:
    """Return relative paths of all files in the run directory."""
    lines: list[str] = []
    if not run_dir.is_dir():
        return lines

    for p in sorted(run_dir.rglob("*"), key=lambda x: str(x)):
        if p.is_file():
            rel = p.relative_to(run_dir)
            lines.append(str(rel))
    return lines
