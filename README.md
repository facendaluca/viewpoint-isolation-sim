# Viewpoint Isolation Simulation (BSc Computer Science FYP)

Simulation framework to study how engagement-optimised short-form video recommendation can
produce stance-distance convergence (“viewpoint isolation”) and operational lock-in over time.

## Project summary
This project builds a controlled simulation with:
- A video corpus (topic, sentiment, viewpoint, duration, tags)
- A user agent with survey-derived phenotypes (Watcher / Sampler / Avoider) 
- A recommender that ranks content using weighted engagement objectives 
- A feedback loop that updates user state and logs the Viewpoint Isolation Index (VII) per step  
## Research design (high level)
- Baseline: deterministic heuristic agent for control and reproducibility 
- Comparative: LLM-based agent to test semantic resilience under identical exposure conditions 
- Outputs: VII time series + action distribution (Avoid/Sample/Watch) and lock-in metrics  

## Repo structure
- `src/fyp_sim/` : core simulation code (models, metrics, policy, engagement, taxonomy)
- `tests/`       : unit tests
- `scripts/`     : small runnable scripts (sanity / experiment runners)
- `configs/`     : experiment configs (planned)
- `outputs/`     : generated results (gitignored)

## Development setup (macOS)
### Requirements
- Python 3.12+

### Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install pytest ruff pre-commit
pre-commit install