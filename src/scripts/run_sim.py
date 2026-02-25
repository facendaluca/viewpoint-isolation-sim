from __future__ import annotations

import csv
import random
from pathlib import Path

from fyp_sim.cli import run_cli
from fyp_sim.models import User, UserPhenotype, Video
from fyp_sim.simulation.engine import run_simulation


def main() -> None:
    rng = random.Random(0)

    user = User(
        phenotype=UserPhenotype.SAMPLER,
        viewpoint_score=0.4,
        interest_vector={"politics": 0.2, "meme": 0.9},
        sentiment_threshold=-0.2,
    )

    video_pool = [
        Video(1, "politics", 0.8, 0.0, 25, tags=("meme",)),
        Video(2, "sports", 0.2, 0.1, 20, tags=("football",)),
        Video(3, "health_wellness", 0.5, 0.2, 39, tags=("sleep",)),
        # NEW: another high-interest candidate via meme tag, but different viewpoint
        Video(4, "politics", -0.6, 0.0, 25, tags=("meme",)),
    ]
    logs = run_simulation(
        user=user,
        video_pool=video_pool,
        steps=20,
        rng=rng,
        enable_interest_updates=True,
        interest_topic_alpha=0.10,
        interest_tag_alpha=0.05,
        interest_decay=0.02,
        interest_normalise=True,
        interest_prune_below=0.001,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "run_log.csv"

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "video_id",
                "action",
                "watch_time_s",
                "interest",
                "vii_t",
                "vii_cum",
                "topic_interest",
                "interest_keys",
            ]
        )
        for row in logs:
            w.writerow(
                [
                    row.t,
                    row.video_id,
                    row.action,
                    row.watch_time_s,
                    f"{row.interest:.4f}",
                    f"{row.vii_t:.4f}",
                    f"{row.vii_cum:.4f}",
                    f"{row.topic_interest:.4f}",
                    row.interest_keys,
                ]
            )
    print(f"Simulation log written to: {out_path}")


if __name__ == "__main__":
    run_cli(main)
