# Viewpoint Isolation Simulation (BSc Computer Science FYP)

Simulation framework to study how engagement-optimised short-form video recommendation can
produce stance-distance convergence (“viewpoint isolation”) and **operational lock-in** over time.

## Project summary
This project builds a controlled simulation with:
- A video corpus (topic, sentiment, viewpoint, duration, free-form tags)
- A user agent with survey-derived phenotypes (**Watcher / Sampler / Avoider**) 
- A recommender that ranks content using weighted engagement objectives (interest + engagement proxy)
- A feedback loop that logs:
  - **VII_t**: per-step viewpoint distance
  - **VII_cum**: running mean of viewpoint distance
  - lock-in metrics (threshold + persistence window) 
  
## Research design (high level)
- **Baseline:** deterministic heuristic agent for control and reproducibility 
- **Sweeps:** LLM-based agent to test semantic resilience under identical exposure conditions 
- **Outputs:** per-run logs (CSV) + aggegated summaries + plots for the report

## Repo structure
- `src/fyp_sim/` : core simulation code (models, metrics, policy, engagement, taxonomy, analysis, simulation engine)
- `src/scripts/`: runnable scripts (batch runners, sweeps, plotting)
- `configs/`     : experiment configs (JSON)
- `results/`     : tracked summaries (CSV)
- `outputs/`     : generated artifacts (gitignored: per-run CSV logs, plots)
- `tests/`       : unit tests

## Development setup (macOS)
### Requirements
- Python **3.12+**
- git

### Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install project (editable)
pip install -e .

# dev tools
pip install pytest ruff pre-commit
pre-commit install

## Synthetic Corpus Generation
To generate and inspect a deterministic corpus:
```bash
python3 -m src.scripts.generate_corpus --config configs/experiment_generated_corpus.json --out outputs/corpus.json
```

To use a generated corpus in an experiment, ensure your config has:
```json
"corpus": {
    "source": "generated",
    "n_videos": 1000,
    "seed": 42,
    "generator": { ... }
}
```
The corpus is deterministically generated based on `N`, `seed`, and `config` settings.

## Structured Batch Execution
The `run_batch.py` script now supports structured outputs and manifest generation.

### Usage
```bash
# Default (static corpus from config)
python3 -m src.scripts.run_batch configs/experiment_baseline.json

# Override with generated corpus (deterministic)
python3 -m src.scripts.run_batch configs/experiment_baseline.json \
  --corpus-mode generated \
  --corpus-size 5000 \
  --corpus-seed 12345
```

### Output Layout
Outputs are saved to `outputs/runs/<RUN_ID>/`:
- `manifest.json`: Metadata, status, and configuration snapshot.
- `resolved_config.json`: The exact configuration used.
- `batch.log`: High-level execution log.
- `runs/run_XXX/`: Per-seed simulation logs (`run.log.csv`).
- `aggregate/summary.csv`: Summary metrics for all runs in the batch.

### Running Sweeps
The `run_sweep.py` script also supports structured outputs and corpus generation.
```bash
# Default sweep (static corpus)
python3 -m src.scripts.run_sweep configs/experiment_sweep.json

# Sweep with generated corpus
python3 -m src.scripts.run_sweep configs/experiment_sweep.json \
  --corpus-mode generated \
  --corpus-size 1000 \
  --corpus-seed 42 \
  --name "my_sweep"
```
Sweep outputs are saved to `outputs/runs/<SWEEP_RUN_ID>/` with an `aggregate/sweep_summary.csv`.

### Plotting
To generate plots from a sweep, point `make_plots.py` to the summary CSV or Run ID:

```bash
# Using explicit summary path
python3 -m src.scripts.make_plots \
  --sweep-summary outputs/runs/<SWEEP_RUN_ID>/aggregate/sweep_summary.csv \
  --out-dir outputs/plots/<SWEEP_RUN_ID>

# Using Run ID (auto-derives paths)
python3 -m src.scripts.make_plots --run-id <SWEEP_RUN_ID>
```