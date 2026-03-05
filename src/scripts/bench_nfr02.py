from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from fyp_sim.analysis import summarise_logs
from fyp_sim.artefacts import _fail_fast_old_alpha, create_run_artefacts
from fyp_sim.benchmarks.common import BenchmarkParams, ConsoleCapture, TimingStats, format_os_text
from fyp_sim.benchmarks.phase_timing import PhaseTimer, build_perf_breakdown
from fyp_sim.benchmarks.system_info import collect_machine_specs
from fyp_sim.corpus import build_corpus
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.runners.csv_io import write_run_log_csv, write_summary_csv
from fyp_sim.simulation.engine import StepLog, run_simulation
from fyp_sim.taxonomy import TOPIC_CATEGORIES

_TOPIC_WEIGHTS: dict[str, float] = {
    "comedy_memes": 0.30,
    "fashion_beauty": 0.17,
    "diy_life_hacks": 0.17,
    "health_wellness": 0.16,
    "politics": 0.04,
    "social_issues_activism": 0.03,
    "international_news": 0.03,
    "finance_economics": 0.05,
    "sports": 0.05,
}

_USER_INTEREST: dict[str, float] = {
    "comedy_memes": 0.80,
    "health_wellness": 0.65,
    "diy_life_hacks": 0.55,
    "fashion_beauty": 0.45,
    "finance_economics": 0.30,
    "sports": 0.25,
    "politics": 0.20,
    "international_news": 0.20,
    "social_issues_activism": 0.20,
}


def _filter_known_topics(weights: dict[str, float]) -> dict[str, float]:
    allowed = set(TOPIC_CATEGORIES)
    return {k: float(v) for k, v in weights.items() if k in allowed}


def build_benchmark_config(params: BenchmarkParams) -> dict[str, Any]:
    """Resolved config snapshot stored with the run artefacts (reproducibility evidence)."""
    topic_weights = _filter_known_topics(_TOPIC_WEIGHTS)
    user_interest = _filter_known_topics(_USER_INTEREST)

    return {
        "scenario": "benchmark_nfr02",
        "steps": int(params.steps),
        "top_k": int(params.top_k),
        "rank_alpha": float(params.rank_alpha),
        # Keep these so summary.csv stays comparable to standard runs
        "lock_in_threshold": 0.10,
        "persistence_window": 10,
        # Benchmark the deterministic heuristic path (engine throughput)
        "policy": {"mode": "heuristic"},
        "enable_interest_updates": False,
        "enable_viewpoint_drift": False,
        "drift_alpha": 0.0,
        "user": {
            "phenotype": "watcher",
            "viewpoint_score": 0.50,
            "interest_vector": user_interest,
            "sentiment_threshold": 0.0,
        },
        "corpus": {
            "source": "generated",
            "n_videos": int(params.n_videos),
            "seed": int(params.seed),
            "generator": {"topic": {"weights": topic_weights}},
        },
    }


def build_user(cfg: dict[str, Any]) -> User:
    u = cfg["user"]
    return User(
        phenotype=UserPhenotype.WATCHER,
        viewpoint_score=float(u["viewpoint_score"]),
        interest_vector={str(k): float(v) for k, v in (u.get("interest_vector", {})).items()},
        sentiment_threshold=float(u["sentiment_threshold"]),
    )


def build_video_pool(cfg: dict[str, Any], *, expected_n: int) -> list[Any]:
    pool = build_corpus(cfg)
    if len(pool) != expected_n:
        raise ValueError(f"Expected n_videos={expected_n}, got {len(pool)}")
    return pool


def print_header(
    console: ConsoleCapture,
    *,
    params: BenchmarkParams,
    run_dir: Path,
    specs: dict[str, Any],
) -> None:
    console.emit("=== NFR-02 Performance Benchmark ===")
    console.emit(
        f"Target: {params.steps} steps with corpus N={params.n_videos} in < {params.threshold_s:.3f}s "
        f"(mean over {params.repeats} runs)"
    )
    console.emit(f"Run dir: {run_dir}")
    console.emit()
    console.emit("--- Machine specs ---")
    console.emit(f"OS: {format_os_text(specs)}")
    console.emit(f"CPU: {specs.get('cpu_brand')}")
    console.emit(f"Logical CPUs: {specs.get('cpu_count_logical')}")
    if "ram_total_gb" in specs:
        console.emit(f"RAM: {specs['ram_total_gb']} GB")
    console.emit(f"Python: {specs.get('python_implementation')} {specs.get('python_version')}")
    console.emit()


def print_results(console: ConsoleCapture, stats: TimingStats) -> None:
    console.emit()
    console.emit("--- Results ---")
    console.emit(f"Mean:  {stats.mean_s:.6f}s")
    console.emit(f"Best:  {stats.best_s:.6f}s")
    console.emit(f"Worst: {stats.worst_s:.6f}s")
    console.emit(f"Status: {'PASS' if stats.passed else 'FAIL'}")


def run_trials(
    params: BenchmarkParams,
    *,
    cfg: dict[str, Any],
    pool: list[Video],
    console: ConsoleCapture,
) -> tuple[list[StepLog], TimingStats]:
    timings: list[float] = []
    logs_first: list[StepLog] | None = None

    for i in range(1, params.repeats + 1):
        user = build_user(cfg)
        rng = random.Random(params.seed)

        t0 = time.perf_counter()
        logs = run_simulation(
            user=user,
            video_pool=pool,
            steps=params.steps,
            rng=rng,
            top_k=params.top_k,
            rank_alpha=float(params.rank_alpha),
            drift_alpha=0.0,
            enable_interest_updates=False,
            enable_viewpoint_drift=False,
            viewpoint_drift_rate=0.0,
        )
        dt = time.perf_counter() - t0

        if logs_first is None:
            logs_first = logs

        timings.append(float(dt))
        console.emit(f"Run {i}/{params.repeats}: {dt:.6f}s")

    assert logs_first is not None
    stats = TimingStats.from_timings(timings, threshold_s=float(params.threshold_s))
    return logs_first, stats


def write_outputs(
    *,
    artefacts: Any,
    cfg: dict[str, Any],
    seed: int,
    logs: list[StepLog],
    specs: dict[str, Any],
    params: BenchmarkParams,
    stats: TimingStats,
    console: ConsoleCapture,
    phase_timer: PhaseTimer | None = None,
) -> None:
    export_t0 = time.perf_counter()

    seed_dir = artefacts.seeds_dir / f"s{seed:05d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    write_run_log_csv(seed_dir / "run_log.csv", logs, include_viewpoint=False)

    summary = summarise_logs(
        logs,
        lock_in_threshold=float(cfg["lock_in_threshold"]),
        persistence_window=int(cfg["persistence_window"]),
    )

    row: dict[str, Any] = {
        "seed": int(seed),
        "steps": int(params.steps),
        "top_k": int(params.top_k),
        "rank_alpha": float(params.rank_alpha),
        "drift_alpha": 0.0,
        **summary,
    }
    write_summary_csv(artefacts.summary_path, [row])

    manifest = json.loads(artefacts.manifest_path.read_text(encoding="utf-8"))
    manifest["machine"] = specs
    manifest["benchmark"] = {
        "requirement_id": "NFR-02",
        "threshold_s": float(params.threshold_s),
        "steps": int(params.steps),
        "n_videos": int(params.n_videos),
        "repeats": int(params.repeats),
        "timings_s": [round(x, 6) for x in stats.timings_s],
        "mean_s": round(stats.mean_s, 6),
        "best_s": round(stats.best_s, 6),
        "worst_s": round(stats.worst_s, 6),
        "pass": bool(stats.passed),
        "timed_section": (
            "run_simulation() only (includes in-memory per-step logging). "
            "Corpus generation + file writes happen outside the timed section."
        ),
    }
    artefacts.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artefacts.root_dir / "stdout.txt").write_text(console.to_text(), encoding="utf-8")

    export_dt = time.perf_counter() - export_t0
    if phase_timer is not None:
        breakdown = build_perf_breakdown(
            timings=phase_timer.timings,
            steps=int(params.steps),
            n_videos=int(params.n_videos),
            top_k=int(params.top_k),
            rank_alpha=float(params.rank_alpha),
            extra_phases_s={"export_logs": float(export_dt)},
        )
        (artefacts.root_dir / "perf_breakdown.json").write_text(
            json.dumps(breakdown, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def parse_args() -> tuple[BenchmarkParams, bool]:
    p = argparse.ArgumentParser(
        description="NFR-02 benchmark: 1000 steps with corpus N=1000 under 1s (3 runs)."
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--n-videos", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rank-alpha", type=float, default=0.5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threshold-s", type=float, default=1.0)
    p.add_argument("--outputs-root", type=Path, default=Path("outputs/benchmarks"))
    p.add_argument(
        "--profile-phases",
        action="store_true",
        help="Write perf_breakdown.json with per-phase timings (includes one extra untimed run).",
    )

    args = p.parse_args()
    params = BenchmarkParams(
        steps=int(args.steps),
        n_videos=int(args.n_videos),
        seed=int(args.seed),
        top_k=int(args.top_k),
        rank_alpha=float(args.rank_alpha),
        repeats=int(args.repeats),
        threshold_s=float(args.threshold_s),
        outputs_root=Path(args.outputs_root),
    )
    params.validate()
    return params, bool(args.profile_phases)


def main() -> None:
    params, profile_phases = parse_args()

    cfg = build_benchmark_config(params)
    _fail_fast_old_alpha(cfg, cfg_path=None)

    # Build corpus once (outside timed section) to benchmark simulation loop throughput.
    pool = build_video_pool(cfg, expected_n=params.n_videos)

    artefacts = create_run_artefacts(
        cfg=cfg,
        cfg_path=None,
        mode="bench_nfr02",
        seeds=[params.seed],
        outputs_root=params.outputs_root,
        corpus=pool,
        runner="src.scripts.bench_nfr02",
        write_config_snapshot=True,
    )

    console = ConsoleCapture()
    specs = collect_machine_specs()
    print_header(console, params=params, run_dir=artefacts.root_dir, specs=specs)

    phase_timer: PhaseTimer | None = None
    if profile_phases:
        console.emit("--- Phase profiling ---")
        phase_timer = PhaseTimer()
        user = build_user(cfg)
        rng = random.Random(params.seed)
        _ = run_simulation(
            user=user,
            video_pool=pool,
            steps=params.steps,
            rng=rng,
            top_k=params.top_k,
            rank_alpha=float(params.rank_alpha),
            drift_alpha=0.0,
            enable_interest_updates=False,
            enable_viewpoint_drift=False,
            viewpoint_drift_rate=0.0,
            phase_tracer=phase_timer,
        )
    logs_first, stats = run_trials(params, cfg=cfg, pool=pool, console=console)
    print_results(console, stats)

    write_outputs(
        artefacts=artefacts,
        cfg=cfg,
        seed=params.seed,
        logs=logs_first,
        specs=specs,
        params=params,
        stats=stats,
        console=console,
        phase_timer=phase_timer,
    )


if __name__ == "__main__":
    main()
