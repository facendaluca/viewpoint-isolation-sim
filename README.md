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