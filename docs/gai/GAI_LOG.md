# GAI Usage Log (Antigravity Agents / ChatGPT)

This log documents any use of Generative AI (GAI) tools during development.
Goal: keep a clear audit trail showing that I remained the primary author, applied critical judgement,and verified all outputs (tests, linting, reproducibility checks).

---

## Project-level details

- **Project:** Viewpoint Isolation Simulation (Short-form video recommender simulation)
- **Repo:** viewpoint-isolation-sim
- **Primary language:** Python
- **GAI tools used:** Antigravity Agents, ChatGPT
- **GAI policy**
  - Allowed: scaffolding, debugging assistance, refactor suggestions, test ideas, docstring drafts, plotting templates
  - Allowed: “explain/critique” of my ideas and code (I decide and implement)
  - Not allowed: inventing results, datasets, citations, or claiming experiments ran when they didn’t
  - Not allowed: “report writing” in an AI voice (I write; AI can help review/structure)
- **Verification policy**
  - `pytest -q`
  - `pre-commit run --all-files`
  - A minimal runtime check (e.g., `python -m scripts.run_batch` / `python -m scripts.run_sweep`)
- **Where evidence lives**
  - Screenshots/exports: `docs/gai/evidence/`
  - Optional diffs/patches: `docs/gai/diffs/`

---

### Entry ID: GAI-2026-02-01-001
- **Feature / change title:** Milestone 6 — plotting pipeline for sweep outputs (heatmaps)
- **Branch:** main
- **PR / Commit(s):** (fill in)
- **Files touched:** src/scripts/make_plots.py, outputs/plots/* (generated)
- **Goal (1–2 lines):** Generate publication-ready plots from `results/sweep_summary.csv` to support analysis and reporting of sweep behaviour (VII and lock-in metrics).
- **GAI used:** Yes, planning only (no direct code merged without review).
- **Constraints I gave the agent:**
    - Plots must be reproducible from commited CSV.
    - Keep the plotting step as a seperate script (no effect on simulation logic).
    - Prefer simple dependencies and a workflow suitable for an academic report.
- **Prompts/instructions (summary):**
    - Asked GAI what tools/libraries are appropriate for producing "pretty plots" from a CSV sweep table and what plots would best communicate results.
    - Asked for a minimal workflow and recommend plot types.
- **Agent output summary (what it produced):**
    - Suggested using a lightweight pipeline: read CSV + plot with Matplotlib (and potentially avoid additional dependencies).
- **My critical evaluation (what I accepted/rejected and why):**
    - Accepted: The idea of keeping plotting isolated in a seperate script, reading from results/sweep_summary.csv, and generating figure files for the report.
    - Rejected: I chose pandas + matplotlib + seaborn rather than raw csv + matplotlib.
      - Reason:
        - This matches the approach taught in my Machine Learning module (Programming 3), where pandas is used for data manipulation and seaborn is used for clearer statistical visualisations built on Matplotlib.
        - pandas also makes pivoting and aggregation (e.g., top_k x alpha grids) clearer and less error-prone than manual CSV parsing.
- **Risk/edge cases noticed:**
    - Ensure scripts do not commit generated artifacts (outputs/plots/ should be gitignored).
    - Ensure the plotting script fails clearly if results/sweep_summary.csv is missing.
- **My edits after AI output:**
    - Implemented plotting with pandas DataFrames and seaborn heatmaps.
    - Defined explicit output directory structure (outputs/plots/)
    - Ensured plots are generated deterministically from CSV inputs (no randomness).
- **Impact / outcome:**
    - Produced clear heatmaps showing how mean VII and lock-in rate vary across sweep parameters (top_k, alpha), suitable for insertion into the final report.
- **Attribution statement:**
    - GAI was used to plan plotting approach; the final tool choice (pandas + matplotlib + seaborn) and implementation were decided, written, and validated by me.

---

## Notes for assessors
- I use a dedicated branch for GAI-assisted work (`ai/antigravity`) and merge into `main` after review.
- All GAI-assisted contributions are verified with tests/lint and reviewed for correctness, reproducibility, and scope.
