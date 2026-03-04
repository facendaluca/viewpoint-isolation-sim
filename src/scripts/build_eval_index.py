from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DESCRIPTIONS: dict[str, str] = {
    "E1_baseline_single_watcher.json": "RQ1 baseline variability: single Watcher, fixed corpus + fixed objective.",
    "E2_baseline_multi_phenotype_cohort.json": "Phenotype differences: Watcher vs Sampler vs Avoider in same run (agent_id).",
    "E3_explore_low.json": "RQ3 exploration sensitivity: low curiosity (0.05).",
    "E4_explore_base.json": "RQ3 exploration sensitivity: baseline curiosity (0.30).",
    "E5_explore_high.json": "RQ3 exploration sensitivity: high curiosity (0.60).",
    "E6_sentiment_strict_vs_lenient.json": "Mood regulation sensitivity: strict vs lenient sentiment thresholds in one cohort run.",
}


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for date_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        for run_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
            if (run_dir / "manifest.json").exists():
                out.append(run_dir)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Build outputs/runs/index.md for configs/eval runs.")
    p.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    p.add_argument("--out", type=Path, default=Path("outputs/runs/index.md"))
    args = p.parse_args()

    runs = []
    for run_dir in _iter_run_dirs(args.runs_root):
        manifest = _load_json(run_dir / "manifest.json")
        cfg_path = (manifest.get("cfg_path") or "").replace("\\", "/")
        if "configs/eval/" not in cfg_path:
            continue

        cfg_name = Path(cfg_path).name
        resolved_cfg = (
            _load_json(run_dir / "config_resolved.json")
            if (run_dir / "config_resolved.json").exists()
            else {}
        )
        n_agents = int(resolved_cfg.get("n_agents", 1))

        corpus = manifest.get("corpus", {}) or {}
        runs.append(
            {
                "cfg_name": cfg_name,
                "run_id": manifest.get("run_id", run_dir.name),
                "date_ymd": manifest.get("date_ymd", run_dir.parent.name),
                "n_agents": n_agents,
                "seeds": manifest.get("seeds", []),
                "corpus_hash": corpus.get("hash", ""),
                "notes": DESCRIPTIONS.get(cfg_name, ""),
            }
        )

    runs.sort(key=lambda r: r["cfg_name"])

    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Evaluation runs index\n")
    if not runs:
        lines.append("No configs/eval runs found under outputs/runs.\n")
    else:
        lines.append("| Config | Run ID | Date | Agents | Seeds | Corpus hash | Notes |\n")
        lines.append("|--------|--------|------|--------|-------|-------------|-------|\n")
        for r in runs:
            seeds_str = ",".join(str(x) for x in r["seeds"])
            lines.append(
                f"| `{r['cfg_name']}` | `{r['run_id']}` | {r['date_ymd']} | {r['n_agents']} | `{seeds_str}` | `{r['corpus_hash'][:12]}` | {r['notes']} |\n"
            )

        args.out.write_text("".join(lines), encoding="utf-8")
        print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
