"""
Script to generate and inspect a video corpus deterministically.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from fyp_sim.corpus import build_corpus


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and inspect a video corpus.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config JSON.",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        help="Override number of videos to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the corpus as JSON.",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)

    # Apply overrides
    if args.n_videos is not None:
        # Assuming config structure: "corpus": { "n_videos": ... }
        if "corpus" not in cfg:
            cfg["corpus"] = {}
        cfg["corpus"]["n_videos"] = args.n_videos
        print(f"Overriding n_videos: {args.n_videos}")

    if args.seed is not None:
        # Assuming config structure: "corpus": { "seed": ... }
        if "corpus" not in cfg:
            cfg["corpus"] = {}
        cfg["corpus"]["seed"] = args.seed
        print(f"Overriding seed: {args.seed}")

    try:
        videos = build_corpus(cfg)
    except Exception as e:
        print(f"Error building corpus: {e}", file=sys.stderr)
        return 1

    # Print summary
    n = len(videos)
    print(f"Generated {n} videos.")

    if n > 0:
        topics = {}
        sentiments = {}
        viewpoints = {}
        durations = []
        tags = set()

        for v in videos:
            topics[v.topic_category] = topics.get(v.topic_category, 0) + 1
            sentiments[v.sentiment_score] = sentiments.get(v.sentiment_score, 0) + 1
            viewpoints[v.viewpoint_score] = viewpoints.get(v.viewpoint_score, 0) + 1
            durations.append(v.duration_s)
            tags.update(v.tags)

        print("\n--- Summary ---")
        print("Topics:")
        for t, count in sorted(topics.items()):
            print(f"  {t}: {count} ({count / n:.1%})")

        print("\nSentiments:")
        for s, count in sorted(sentiments.items()):
            print(f"  {s}: {count} ({count / n:.1%})")

        print("\nViewpoints:")
        for v, count in sorted(viewpoints.items()):
            print(f"  {v}: {count} ({count / n:.1%})")

        print("\nDurations:")
        print(f"  Min: {min(durations)}s")
        print(f"  Max: {max(durations)}s")
        print(f"  Avg: {sum(durations) / n:.1f}s")

        print(f"\nUnique Tags: {len(tags)}")

    # Write output if requested
    if args.out:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize videos to dicts
        data = [dataclasses.asdict(v) for v in videos]

        with out_path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"\nWrote corpus to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
