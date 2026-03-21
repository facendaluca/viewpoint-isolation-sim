# Viewpoint Isolation Simulation (BSc Computer Science FYP)

Simulation framework to study how engagement-optimised short-form video recommendation can
produce stance-distance convergence (“viewpoint isolation”) and **operational lock-in** over time.

## Project overview
This project builds a controlled simulation to investigate how engagement-optimised short-form video recommendation impacts user content diversity over time. Specifically, it studies the phenomena of "viewpoint isolation" and implicit feedback loops, where the recommender's objectives algorithmically limit topic exposure. By modelling user behaviour via qualitative phenotypes (Watcher, Sampler, Avoider) interacting with a synthetic video corpus, the simulation explores how algorithmic design choices can unintentionally lock users into narrow ideological silos.

## Key concepts
- **Viewpoint isolation**: A quantitative metric representing the constraint or convergence of the political or cultural viewpoints a user is continually exposed to over time.
- **Interest vector**: A dynamic mathematical representation of a user's topic affinities and tag preferences, which are updated algorithmically based on watch time duration.
- **Implicit feedback loops**: A self-reinforcing cycle where a user's engagement informs the recommender systems perceived interest, prompting the continuous serving of similar content. This restricts exploration and establishes operational lock-in.

## Repository structure
```text
├── configs/   # Experiment configurations (JSON)
├── outputs/   # Generated run artifacts (logs, summaries, plots)
├── pages/     # Streamlit dashboard pages
├── results/   # Legacy outputs location
├── src/       # Core simulation and scripts
├── tests/     # Unit and integration tests
├── ui/        # UI components for the dashboard
└── Dashboard.py # Streamlit UI entrypoint
```

## Running experiments
First, set up a Python 3.12+ virtual environment and install dependencies:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff pre-commit
pre-commit install
```

Run experiments via the module scripts. Replace config paths with any of the files in `configs/`:
- **Run a baseline batch**: `python -m src.scripts.run_batch configs/experiment_baseline.json`
- **Run a parameter sweep**: `python -m src.scripts.run_sweep configs/experiment_sweep.json`

To generate plots for a specific run directory:
```bash
python -m src.scripts.make_plots --run-dir outputs/runs/YYYYMMDD/<run_id>
```
Plots can also be generated from the Streamlit dashboard: open **Explore Results**, select a run directory, then use the **Plots** tab to generate or regenerate plot outputs for that run.

## Dashboard / UI
An interactive Streamlit dashboard is available to run and explore experiments. Launch it via:
```bash
streamlit run Dashboard.py
```

## Experiment artefacts
Experiments write out to an automatically generated directory structure under `outputs/runs/`, following the format `YYYYMMDD/<HHMMSS>Z_<mode>_<hash8>/`. The `run_id` encodes the time, simulation mode (e.g., `baseline`, `sweep`), and an 8-character hash of the resolved configuration snapshot.

A typical run directory structure:
```text
outputs/runs/20260223/153022Z_baseline_a1b2c3d4/
├── config_resolved.json   # Snapshot of the full run config
├── manifest.json          # High-level metadata (seeds, key params)
├── summary.csv            # Aggregated metrics across all seeds
├── plots/                 # Generated plots (empty until plots are generated via CLI or dashboard)
└── seeds/
    ├── s00042/
    │   └── run_log.csv    # Step-by-step engagement log for seed
    └── ...
```

## Testing
Run the test suite using `pytest`:
```bash
pytest
```
*This executes unit and integration tests located in `tests/`.*

## Benchmarks
Performance benchmarks are included in `src/scripts/bench_nfr02.py`. Execute it to measure simulation overhead:
```bash
python -m src.scripts.bench_nfr02
```

## LLM mode
The simulation supports an "LLM mode" where a large language model replaces the deterministic heuristic user policy, testing semantic resilience and content decisions under identical exposure conditions. The decision-making strictly adheres to a predefined JSON schema containing actions such as 'Watch', 'Avoid', or 'Sample'.
LLM mode requires an OpenAI-compatible endpoint and is configured within the `policy.llm` section of the JSON configurations (`model`, `base_url`).  
*Caution: LLM availability and network latency can cause timeouts; the simulation includes a rigid fallback to the heuristic decider.*

**Hardware note**: LLM mode experiments for this project were executed on a Windows 11 PC equipped with an AMD Ryzen 9 processor, 64GB DDR5 RAM, and an NVIDIA RTX 3090 GPU.

## Reproducibility notes
- **Deterministic RNG**: The simulation is fully deterministic and reproducible given a random seed state.
- **Config snapshots**: A stable configuration snapshot (`config_resolved.json`) is saved with every recorded artefact.

## Limitations / Future work
- **Simplified phenotypes**: The user agent phenotypes (Watcher, Sampler, Avoider) are heuristic and may not fully capture nuanced, real-world human behavior.
- **Synthetic corpus constraints**: Generating realistic video metadata is difficult, and mapping genuine user engagement signals to a pre-defined set of tags lacks authentic contextual depth.
- **Multi-modal feedback**: Future work could incorporate more complex feedback mechanisms (likes, shares, comments) to improve modeling of systemic lock-in mechanics.