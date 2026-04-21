# Viewpoint Isolation Simulation (BSc Computer Science FYP)

A simulation framework to study how engagement-optimised short-form video recommendation can
produce stance-distance convergence ("viewpoint isolation") and **operational lock-in** over time.

## Project overview

This project builds a controlled simulation to investigate how engagement-optimised short-form video recommendation impacts user content diversity over time. Specifically, it studies the phenomena of "viewpoint isolation" and implicit feedback loops, where the recommender's objectives algorithmically limit topic exposure. User behaviour is modelled based on qualitative phenotypes (Watcher, Sampler, Avoider) interacting with a synthetic video corpus; the simulation explores how algorithmic design choices can unintentionally lock users into narrow ideological silos.

The simulation is mechanism-focused: it does not claim to fully reproduce any commercial platform. Results should be interpreted as evidence under the model's stated assumptions, not as direct proof of real-world societal causality.

## Key concepts

- **Viewpoint isolation**: A quantitative metric representing the constraint or convergence of the political or cultural viewpoints a user is continually exposed to over time.
- **Viewpoint Isolation Index (VII)**: The main temporal signal for viewpoint isolation. VII summarises stance distance over time: lower stance distance means consumed content is closer to the user's current viewpoint.
- **Interest vector**: A dynamic mathematical representation of a user's topic affinities and tag preferences, which are updated algorithmically based on watch time duration.
- **Implicit feedback loops**: A self-reinforcing cycle where a user's engagement informs the recommender system's perceived interest, prompting the continuous serving of similar content. This restricts exploration and establishes operational lock-in.
- **Lock-in**: A sustained state where the user's consumed content stays below a stance-distance threshold for a configurable number of consecutive steps. Detected using `lock_in_threshold` and `persistence_window`.
- **Phenotypes**: Heuristic behavioural profiles (Watcher, Sampler, Avoider) that model distinct patterns of user engagement with recommended content.

## Repository structure

```text
├── configs/              # Experiment configurations (JSON)
│   └── eval/             # Dissertation-aligned evaluation presets (E1–E6)
├── docs/                 # Documentation and GAI log
├── outputs/              # Generated run artefacts (logs, summaries, plots)
├── pages/                # Streamlit dashboard pages
│   ├── 2_Run_Locally.py
│   ├── 3_Explore_Results.py
│   ├── 4_About.py
│   └── 5_Compare_Runs.py
├── src/
│   ├── fyp_sim/          # Core simulation package
│   │   ├── agents/       # Agent decision logic (heuristic + LLM)
│   │   ├── benchmarks/   # Performance benchmarking utilities
│   │   ├── corpus/       # Synthetic video corpus generation and loading
│   │   ├── examiner_dashboard/  # Dashboard backend (config resolution, run execution)
│   │   ├── llm/          # LLM mode prompts, schemas, and contracts
│   │   ├── plotting/     # Figure generation pipeline (single-run, multi-agent, multi-run, compare)
│   │   ├── runners/      # Seed sweep and batch execution
│   │   └── simulation/   # Engine, viewpoint drift, and optimised engine variant
│   └── scripts/          # CLI entrypoints (run_batch, run_sweep, make_plots, bench_nfr02, etc.)
├── tests/                # Unit and integration tests
├── ui/                   # UI components, state management, and form processing
│   └── views/            # Page-level view renderers and section components
├── Dashboard.py          # Streamlit entrypoint
└── pyproject.toml        # Project metadata and tool configuration
```

## Setup

Requires **Python 3.11+** (developed and tested on 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development (linting, testing, pre-commit hooks):

```bash
pip install pytest ruff pre-commit
pre-commit install
```

## Running the dashboard

The primary way to interact with the project is through the Streamlit examiner dashboard:

```bash
streamlit run Dashboard.py
```

The dashboard has four pages accessible from the sidebar:

| Page | Purpose |
|------|---------|
| **Overview** | Examiner-facing guide to the evaluation workflow, parameters, result interpretation, and artefact structure |
| **Run Locally** | Create bounded heuristic simulation runs from dissertation-aligned presets |
| **Explore Results** | Inspect run artefacts, generate figures, and browse outputs |
| **Compare Runs** | Side-by-side comparison of two completed runs |

There is also an **About** page with brief design notes.

### Recommended examiner journey

1. Open **Run Locally** and choose a preset scenario.
2. Optionally adjust bounded parameters or add advanced JSON overrides.
3. Run the simulation and wait for the run directory to be created.
4. Open **Explore Results** and select the generated run.
5. Use the **Plots** tab to generate figures for that run.
6. Review the **Overview**, **Config**, **Manifest**, **Summary**, **Seeds**, **Plots**, and **Files** tabs.
7. Use **Compare Runs** after generating at least two runs.

---

## Run Locally

The Run Locally page creates bounded local heuristic simulation runs. It is the main entry point for generating experimental data.

### Preset scenarios

Each run starts from one of six dissertation-aligned evaluation presets (E1–E6). These correspond to the core evaluation conditions used in the dissertation:

| Preset | Condition | What it tests |
|--------|-----------|---------------|
| **E1** – Baseline single watcher | Single heuristic watcher, fixed ranking | Baseline convergence and lock-in under standard conditions |
| **E2** – Baseline multi-phenotype cohort | Watcher + Sampler + Avoider under same environment | Phenotype-level differences in VII trajectory and lock-in |
| **E3** – Low exploration | Curiosity = 0.05 | Whether reduced exploration accelerates narrowing |
| **E4** – Base exploration | Curiosity = 0.30 | Midpoint reference for the exploration sweep |
| **E5** – High exploration | Curiosity = 0.60 | Whether stronger exploration delays or weakens lock-in |
| **E6** – Sentiment strict vs lenient | Two watchers with different sentiment thresholds | Whether sentiment filtering changes convergence behaviour |

Presets set sensible defaults for all parameters. Select a preset from the dropdown to load its configuration.

### Bounded parameter controls

After selecting a preset, six parameters are exposed as bounded UI controls:

| Parameter | Range | Default | What it controls |
|-----------|-------|---------|------------------|
| `steps` | 25–500 | 200 | Number of interaction timesteps per run. Longer runs give more time for convergence or lock-in to develop. |
| `top_k` | 1–10 | 5 | Number of candidate videos considered at each recommendation step. Larger pools widen the selection before the final choice. |
| `rank_alpha` | 0.00–1.00 | 0.30 | Weighting between interest and engagement during ranking. Higher values increase the influence of the current ranking signal. |
| `drift_alpha` | 0.00–0.20 | 0.02 | Strength of viewpoint movement after interactions (when drift is enabled). Higher values mean faster viewpoint adjustment. |
| `lock_in_threshold` | 0.00–1.00 | 0.20 | Stance-distance boundary below which a step is considered "locked in". |
| `persistence_window` | 1–50 | 10 | How many consecutive steps must stay at or below the threshold before a lock-in episode is counted. Higher values make lock-in harder to trigger. |

These controls are clamped to safe ranges. Adjusting them changes the run configuration without requiring manual JSON editing.

### Advanced configuration (JSON)

An expandable section below the bounded controls accepts optional JSON overrides. This is for settings not exposed as bounded controls, such as:

- `seeds` – override the seed list (e.g. `{"seeds": [0, 1, 2]}`)
- `enable_viewpoint_drift` – toggle viewpoint updating
- `enable_interest_updates` – toggle adaptive interest updating
- `interest_decay`, `interest_topic_alpha`, `interest_tag_alpha` – fine-tune interest adaptation
- `interest_normalise`, `interest_prune_below` – interest profile maintenance
- `policy.curiosity` – change exploration probability
- `user.phenotype`, `user.viewpoint_score`, `sentiment_threshold` – change agent behaviour

Advanced overrides are merged on top of the preset and bounded controls. The resolved configuration preview (expandable below the controls) shows the exact final config that will be used.

Most users should start from a preset and bounded controls only. Use advanced config when you need fine-grained control over parameters not available through the bounded controls.

### Submitting a run

Once the configuration is valid (no validation errors shown), click **Run heuristic** to execute the simulation. The page shows real-time progress and, on completion, displays a success banner with a link to **Explore Results**.

Validation guards include: range clamping on bounded fields, JSON parse validation, seed list constraints (non-empty, integers only, max 20 values, range 0–99999), and type checking on all overrides.

---

## Explore Results

The Explore Results page is a read-only inspector for completed run directories. Select a run from the dropdown to load it.

The page provides seven tabs:

### Overview

Displays run metadata from the manifest (mode, run ID, date, seeds) and an artefact checklist showing which expected files are present.

### Config

Shows the full `config_resolved.json` snapshot — the exact configuration used for the run. This is the reproducibility anchor. Downloadable.

### Manifest

Displays `manifest.json`, which contains high-level run metadata: run ID, date, mode, seeds, and other context. Downloadable.

### Summary

Renders `summary.csv` as a table. This contains aggregated metrics across the run (per-seed and overall). Downloadable.

### Seeds

Select a seed subdirectory and inspect its `run_log.csv` — the step-by-step engagement log for that individual seed. Each row records what happened at each timestep: which video was recommended, what action the agent took, the viewpoint distance at that step, and the cumulative isolation index. Downloadable.

### Plots

This tab handles figure generation and browsing:

1. **Status summary**: shows whether plots exist, how many image files are present, and whether they are up to date relative to run inputs.
2. **Generation controls**: validates that the run has the required inputs (manifest + either summary or seed logs), then offers a "Generate plots" button. A "Regenerate (overwrite)" checkbox allows re-creating figures.
3. **Figure browser**: after generation, browse figures in single-figure or gallery view. Supports filtering by filename, sorting, navigation (previous/next), and downloading individual images.

### Files

Lists all files in the run directory (up to 250 entries) for a complete view of the artefact tree.

---

## Generated figures

The plotting pipeline produces different figures depending on whether the run is single-agent or multi-agent.

### Single-agent runs (e.g. E1, E3, E4, E5)

| Figure | Filename | What it shows |
|--------|----------|---------------|
| **A – VII Trajectory** | `figure_a_vii_trajectory.png` | Viewpoint distance over time with lock-in threshold band, mean VII, final isolation index, and share of steps at or below threshold |
| **B – Action Distribution** | `figure_b_action_distribution.png` | Proportion of Watch, Sample, and Avoid actions across the run |
| **C – Lock-in Episodes** | `figure_c_lockin_episodes.png` | Timeline of lock-in episodes showing when and how long sustained lock-in occurred |

### Multi-agent runs (e.g. E2, E6)

| Figure | Filename | What it shows |
|--------|----------|---------------|
| **E – Phenotype VII Trajectories** | `figure_e_phenotype_vii_trajectories.png` | Per-phenotype VII trajectories with smoothing, allowing direct comparison across phenotypes |
| **F – Phenotype Action Dynamics** | `figure_f_phenotype_action_dynamics.png` | Action distributions broken down by phenotype |
| **G – Phenotype Lock-in Outcomes** | `figure_g_phenotype_lockin_outcomes.png` | Comparative lock-in metrics across phenotypes |
| **G (sup) – Phenotype Lock-in Timeline** | `figure_g_sup_phenotype_lockin_timeline.png` | Supplementary timeline view of lock-in by phenotype |

### Multi-seed variability (any run with ≥2 seeds)

| Figure | Filename | What it shows |
|--------|----------|---------------|
| **D – Multi-run Variability** | `figure_d_multi_run_variability.png` | Mean VII with confidence intervals across seeds, showing how variable the outcome is under repeated runs |

All figures are saved in both PNG and PDF formats under the run's `plots/` directory.

---

## How to interpret the results

### Reading the evidence chain

Plots are strongest when read alongside the structured artefacts. The intended evidence path is:

1. **Resolved config** (`config_resolved.json`) — confirms exactly what parameters were used.
2. **Manifest** (`manifest.json`) — confirms when the run happened, which seeds were used, and the execution mode.
3. **Summary** (`summary.csv`) — aggregated metrics for quick comparison across seeds.
4. **Seed logs** (`seeds/sXXXXX/run_log.csv`) — step-by-step evidence for individual seeds. Each row contains the step ID, recommended video, agent action, viewpoint distance, and cumulative isolation index.
5. **Figures** (`plots/`) — visual summaries of the above data.

### Interpreting VII

A downward VII trajectory means consumed content is moving closer to the user's viewpoint. The key question is whether the narrowing is brief, gradual, or sustained. A single low point is not sufficient evidence of lock-in — use the threshold-based lock-in metrics to judge stability.

### Interpreting lock-in

Lock-in is defined by two parameters: `lock_in_threshold` (the stance-distance boundary) and `persistence_window` (how many consecutive steps must stay below that boundary). Key metrics to look at:

- **Time to first lock-in** — how quickly the system reaches sustained narrowing
- **Total lock-in steps** — proportion of the run spent in lock-in
- **Episode count** — how many distinct lock-in episodes occurred
- **Consecutive lock-in length** — how entrenched the narrowing becomes

### Seed variability

Do not over-interpret a single seed. Repeated seeds under the same condition show whether a pattern is robust, fragile, or highly variable. Figure D (multi-run variability) is the primary visual for this. The confidence interval width indicates how much the outcome depends on random initialisation.

### Phenotype comparison

In cohort runs (E2, E6), compare phenotype trajectories separately. The evidence question is whether the same environment produces meaningfully different behaviour profiles — not just different averages, but different shapes and lock-in timing.

### Exploration sweep

Compare E3, E4, and E5 together. The hypothesis is that higher exploration delays or weakens lock-in. Look for later onset, fewer lock-in episodes, or weaker sustained lock-in rather than assuming higher exploration should eliminate it entirely.

### Sentiment comparison

In E6, compare strict and lenient agents on both action mix and convergence outcomes. A different VII trajectory is more convincing when matched by a behavioural explanation (e.g. different Watch/Avoid ratios).

---

## Compare Runs

The Compare Runs page allows side-by-side comparison of two completed runs. Select Run A and Run B from the dropdowns.

### Comparison tabs

| Tab | What it shows |
|-----|---------------|
| **VII Trajectory** | Overlaid VII trajectories for both runs on the same axes, each with its own threshold line |
| **Lock-in Timeline** | Side-by-side lock-in episode timelines |
| **Action Mix** | Comparative action distributions |
| **Key Deltas** | Numeric delta panel: final VII, time to first lock-in, total lock-in steps, and watch rate, with directional indicators |

### Interpretation caveats

- If runs use different `lock_in_threshold` values, a warning is displayed and each run's own threshold is shown on the plots.
- Comparing runs with different `steps` counts, seed lists, or phenotype configurations requires care — the comparison is most informative when only one variable differs between the two runs.
- The comparison uses the first seed from each run for trajectory-level plots.

---

## Running experiments via CLI

For scripted or batch usage outside the dashboard:

```bash
# Run a baseline batch
python -m src.scripts.run_batch configs/eval/E1_baseline_single_watcher.json

# Run a parameter sweep
python -m src.scripts.run_sweep configs/experiment_sweep.json

# Generate plots for a specific run directory
python -m src.scripts.make_plots --run-dir outputs/runs/YYYYMMDD/<run_id>
```

Replace config paths with any JSON file under `configs/`.

## Experiment artefacts

Experiments write to an automatically generated directory structure under `outputs/runs/`, following the format `YYYYMMDD/<HHMMSS>Z_<mode>_<hash8>/`. The `run_id` encodes the time, simulation mode (e.g., `baseline`, `sweep`), and an 8-character hash of the resolved configuration snapshot.

```text
outputs/runs/20260223/153022Z_baseline_a1b2c3d4/
├── config_resolved.json   # Full resolved config used for the run
├── manifest.json          # Run metadata and high-level context
├── summary.csv            # Aggregated metrics across all seeds
├── plots/                 # Generated figures (PNG + PDF, empty until generated)
└── seeds/
    └── s00042/
        └── run_log.csv    # Step-by-step engagement log for one seed
```

## Testing

```bash
pytest
```

This executes unit and integration tests located in `tests/`. The test suite covers simulation logic, config validation, artefact integrity, plotting robustness, and dashboard backend/UI form processing.

## Benchmarks

Performance benchmarks are included in `src/scripts/bench_nfr02.py`:

```bash
python -m src.scripts.bench_nfr02
```

## LLM mode

The simulation supports an "LLM mode" where a large language model replaces the deterministic heuristic user policy, testing semantic resilience and content decisions under identical exposure conditions. The decision-making strictly adheres to a predefined JSON schema containing actions such as 'Watch', 'Avoid', or 'Sample'.

LLM mode requires an OpenAI-compatible endpoint and is configured within the `policy.llm` section of the JSON configurations (`model`, `base_url`).

The examiner-facing dashboard is centred on the deterministic heuristic workflow. LLM experiments belong to the dissertation's comparative analysis but are not the main examiner-run path.

*Caution: LLM availability and network latency can cause timeouts; the simulation includes a rigid fallback to the heuristic decider.*

**Hardware note**: LLM mode experiments for this project were executed on a Windows 11 PC equipped with an AMD Ryzen 9 processor, 64 GB DDR5 RAM, and an NVIDIA RTX 3090 GPU.

## Reproducibility notes

- **Deterministic RNG**: The simulation is fully deterministic and reproducible given a random seed state.
- **Config snapshots**: A stable configuration snapshot (`config_resolved.json`) is saved with every recorded artefact, ensuring any run can be re-examined or replicated.
- **Seeded corpus generation**: The synthetic video corpus is generated deterministically from a configurable seed (`corpus.seed`), so the same item pool is used across conditions.

## Limitations / future work

- **Simplified phenotypes**: The user agent phenotypes (Watcher, Sampler, Avoider) are heuristic and may not fully capture nuanced, real-world human behaviour.
- **Synthetic corpus constraints**: Generating realistic video metadata is difficult, and mapping genuine user engagement signals to a pre-defined set of tags lacks authentic contextual depth.
- **Multi-modal feedback**: Future work could incorporate more complex feedback mechanisms (likes, shares, comments) to improve modelling of systemic lock-in mechanics.
- **Scale**: The simulation operates at a small scale (hundreds of steps, thousands of videos) and does not model platform-level dynamics such as creator incentives or network effects.
