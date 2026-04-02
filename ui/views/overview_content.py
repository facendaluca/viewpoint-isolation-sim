# ui/views/overview_content.py
from __future__ import annotations

from typing import Final

PROJECT_SUMMARY: Final[list[str]] = [
    "This dashboard presents a controlled simulation study of short-form video recommendation rather than a live-platform replica.",
    "The project investigates whether engagement-optimised ranking can produce stance-distance convergence and operational lock-in over time.",
    "The simulation uses synthetic video items, survey-grounded heuristic phenotypes, seeded execution, and structured run artefacts for reproducible comparison.",
    "Viewpoint Isolation Index (VII) summarises stance distance over time: lower stance distance means the consumed content is closer to the user's current viewpoint.",
    "The main evidence comes from run artefacts, seed-level logs, summaries, and generated figures that can be inspected in the dashboard.",
]

DASHBOARD_CAPABILITIES: Final[list[str]] = [
    "Create a local heuristic run from a preset scenario, bounded parameter changes, or advanced JSON overrides.",
    "Inspect completed run directories in Explore Results, including their overview, config snapshot, manifest, summary, seeds, plots, and file listing.",
    "Generate or regenerate figures for a selected run from the Plots tab.",
    "Compare two completed runs side-by-side once at least two runs have been generated.",
    "Review and download key artefacts such as config, manifest, summaries, and seed-level logs where available.",
]

PRESET_GUIDE: Final[list[dict[str, str]]] = [
    {
        "name": "E1_baseline_single_watcher",
        "tests": "Baseline mechanism check for one heuristic watcher under fixed ranking conditions.",
        "look_for": "Whether stance distance trends downward across seeds, whether lock-in appears, and how variable the outcome is between repeated runs.",
        "expected": "This is the clearest single-agent reference condition for baseline convergence and lock-in behaviour.",
    },
    {
        "name": "E2_baseline_multi_phenotype_cohort",
        "tests": "Baseline cohort comparison across watcher, sampler, and avoider phenotypes under the same environment.",
        "look_for": "Differences in VII trajectories, action mix, and lock-in behaviour between phenotypes rather than only one overall mean.",
        "expected": "This condition is most useful for examiner-facing phenotype comparison and cohort-level variability.",
    },
    {
        "name": "E3_explore_low",
        "tests": "Low-exploration sensitivity condition with curiosity reduced to 0.05.",
        "look_for": "Earlier narrowing, steeper VII reduction, or faster lock-in relative to the base and high-exploration conditions.",
        "expected": "If Chapter 3's exploration hypothesis holds, this should be the least resistant of the three exploration settings.",
    },
    {
        "name": "E4_explore_base",
        "tests": "Reference exploration condition with curiosity set to 0.30.",
        "look_for": "Use this as the midpoint comparison when interpreting whether lower or higher exploration meaningfully changes convergence and lock-in.",
        "expected": "This should act as the baseline anchor for the exploration sweep rather than the strongest or weakest case.",
    },
    {
        "name": "E5_explore_high",
        "tests": "Higher-exploration sensitivity condition with curiosity increased to 0.60.",
        "look_for": "Delayed convergence, later lock-in, or weaker lock-in intensity relative to E3 and E4.",
        "expected": "The key question is whether lock-in is postponed or softened, not whether it disappears entirely.",
    },
    {
        "name": "E6_sentiment_strict_vs_lenient",
        "tests": "Sentiment-tolerance comparison between two watcher agents with strict and lenient sentiment thresholds.",
        "look_for": "Differences in action distribution, VII trajectory, and lock-in outcomes caused by how readily each agent tolerates negative content.",
        "expected": "This condition helps show whether sentiment filtering changes exposure and convergence behaviour even when the broader ranking setup remains fixed.",
    },
]

PARAMETER_GROUPS: Final[list[dict[str, object]]] = [
    {
        "title": "Run setup and reproducibility",
        "items": [
            (
                "steps",
                "Number of interaction timesteps. Increasing it gives a longer trajectory and more opportunity for convergence or lock-in to emerge.",
            ),
            (
                "top_k",
                "How many candidate items are considered at each step. Larger values widen the candidate pool before the final choice is made.",
            ),
            (
                "seed / seeds",
                "Controls deterministic repeatability. One seed is a single run; multiple seeds are used to judge variability rather than over-reading one trajectory.",
            ),
            (
                "engine",
                "Execution backend for the simulation. Examiners typically keep this fixed unless you are explicitly testing implementation variants.",
            ),
        ],
    },
    {
        "title": "Ranking and convergence dynamics",
        "items": [
            (
                "rank_alpha",
                "Controls how strongly ranking weight influences the recommendation score. Higher values usually make the current ranking signal matter more at each step.",
            ),
            (
                "lock_in_threshold",
                "The stance-distance boundary below which the run is treated as being in lock-in.",
            ),
            (
                "persistence_window",
                "How many consecutive steps must stay at or below the threshold before lock-in is counted. Higher values make lock-in harder to trigger.",
            ),
            (
                "enable_viewpoint_drift",
                "Turns viewpoint updating on or off. With drift enabled, the user's viewpoint can move over time in response to consumed content.",
            ),
            (
                "drift_alpha",
                "Controls the strength of viewpoint movement when drift is enabled. Higher values mean faster viewpoint adjustment.",
            ),
        ],
    },
    {
        "title": "Behaviour and interest adaptation",
        "items": [
            (
                "policy.mode",
                "Agent decision mode. The examiner-facing dashboard centres on the heuristic path.",
            ),
            (
                "policy.curiosity",
                "Exploration probability. Raising it encourages more off-path sampling; lowering it makes the behaviour more reinforcement-driven.",
            ),
            (
                "user.phenotype",
                "Behavioural profile such as watcher, sampler, or avoider. This changes how the agent tends to respond to recommended items.",
            ),
            (
                "user.viewpoint_score",
                "Starting stance position on the viewpoint scale. Changing it alters the user's initial distance from different content items.",
            ),
            (
                "sentiment_threshold",
                "How tolerant the agent is of negative or emotionally heavy content. Higher thresholds make the agent more selective.",
            ),
            (
                "enable_interest_updates",
                "Turns adaptive interest updating on or off across the run.",
            ),
            (
                "interest_topic_alpha",
                "Strength of topic-level interest updates after interactions.",
            ),
            (
                "interest_tag_alpha",
                "Strength of tag-level interest updates after interactions.",
            ),
            (
                "interest_decay",
                "How quickly older interests fade. Higher decay reduces the persistence of earlier interaction history.",
            ),
            (
                "interest_normalise",
                "Whether the interest profile is normalised after updates.",
            ),
            (
                "interest_prune_below",
                "Removes very small interest weights below the chosen cutoff to keep the profile tidy and stable.",
            ),
        ],
    },
    {
        "title": "Multi-agent and corpus controls",
        "items": [
            (
                "n_agents / agents",
                "Defines a cohort experiment instead of a single-user run. Useful when comparing phenotypes or sentiment settings within one environment.",
            ),
            (
                "corpus.n_videos",
                "Size of the generated video pool. Larger corpora generally increase available variety.",
            ),
            (
                "corpus.seed",
                "Deterministic seed for corpus generation so the same underlying pool can be reused across conditions.",
            ),
            (
                "corpus.generator.*",
                "Topic, sentiment, viewpoint, duration, and tag settings shape the structure of the synthetic video pool seen by the agents.",
            ),
        ],
    },
]

RESULT_GUIDE: Final[list[tuple[str, str]]] = [
    (
        "VII over time",
        "VII is the main temporal signal for viewpoint isolation. A downward trajectory means consumed content is moving closer to the user's viewpoint; the key question is whether that narrowing is brief, gradual, or sustained.",
    ),
    (
        "Lock-in outcomes",
        "Use threshold-based lock-in metrics to judge stability, not just one low point. Time-to-first-lock-in, total lock-in steps, episode counts, and consecutive lock-in length help show how entrenched the narrowing becomes.",
    ),
    (
        "Seed variability",
        "Do not over-interpret one run. Repeated seeds show whether a pattern is robust, fragile, or highly variable under the same condition.",
    ),
    (
        "Phenotype comparison",
        "In cohort runs, compare watcher, sampler, and avoider trajectories separately. The key evidence is whether the same environment produces meaningfully different behaviour profiles.",
    ),
    (
        "Exploration sweep",
        "Compare E3, E4, and E5 together. Look for later or weaker lock-in rather than assuming higher exploration should eliminate it completely.",
    ),
    (
        "Sentiment comparison",
        "In E6, compare strict and lenient agents on both action mix and convergence outcomes. A different VII shape is more convincing when it is matched by a behavioural explanation.",
    ),
    (
        "Files and plots together",
        "Plots are strongest when read alongside manifest, resolved config, summary, and seed logs. Use the artefacts to explain why a figure looks the way it does.",
    ),
]

ARTEFACT_TREE: Final[str] = """outputs/runs/YYYYMMDD/<run_id>/
├── config_resolved.json   # Full resolved config used for the run
├── manifest.json          # Run metadata and high-level context
├── summary.csv            # Aggregated metrics across the run
├── plots/                 # Generated figure images for the selected run
└── seeds/
    └── s00042/
        └── run_log.csv    # Step-by-step log for one seed
"""
