# GAI Usage Log (Antigravity Agents / ChatGPT / Claude Code)

This log documents any use of Generative AI (GAI) tools during development.
Goal: keep a clear audit trail showing that I remained the primary author, applied critical judgement, and verified all outputs (tests, linting, reproducibility checks).

---

## Project-level details

- **Project:** Viewpoint Isolation Simulation (Short-form video recommender simulation)
- **Repo:** viewpoint-isolation-sim
- **Primary language:** Python
- **GAI tools used:** Antigravity Agents, ChatGPT, Claude Code
- **GAI policy**
  - Allowed: scaffolding, debugging assistance, refactor suggestions, test ideas, docstring drafts, plotting templates
  - Allowed: "explain/critique" of my ideas and code (I decide and implement)
  - Not allowed: inventing results, datasets, citations, or claiming experiments ran when they didn't
  - Not allowed: "report writing" in an AI voice (I write; AI can help review/structure)
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
- **PR / Commit(s):** a339bb6
- **Files touched:** src/scripts/make_plots.py, outputs/plots/* (generated)
- **Goal (1–2 lines):** Generate publication-ready plots from `results/sweep_summary.csv` to support analysis and reporting of sweep behaviour (VII and lock-in metrics).
- **GAI used:** Yes, planning only (no direct code merged without review).
- **Constraints I gave the agent:**
    - Plots must be reproducible from committed CSV.
    - Keep the plotting step as a separate script (no effect on simulation logic).
    - Prefer simple dependencies and a workflow suitable for an academic report.
- **Prompts/instructions (summary):**
    - Asked GAI what tools/libraries are appropriate for producing "pretty plots" from a CSV sweep table and what plots would best communicate results.
    - Asked for a minimal workflow and recommended plot types.
- **Agent output summary (what it produced):**
    - Suggested using a lightweight pipeline: read CSV + plot with Matplotlib (and potentially avoid additional dependencies).
- **My critical evaluation (what I accepted/rejected and why):**
    - Accepted: The idea of keeping plotting isolated in a separate script, reading from results/sweep_summary.csv, and generating figure files for the report.
    - Rejected: I chose pandas + matplotlib + seaborn rather than raw csv + matplotlib.
      - Reason:
        - This matches the approach taught in my Machine Learning module (Programming 3), where pandas is used for data manipulation and seaborn is used for clearer statistical visualisations built on Matplotlib.
        - pandas also makes pivoting and aggregation (e.g., top_k x alpha grids) clearer and less error-prone than manual CSV parsing.
- **Risk/edge cases noticed:**
    - Ensure scripts do not commit generated artefacts (outputs/plots/ should be gitignored).
    - Ensure the plotting script fails clearly if results/sweep_summary.csv is missing.
- **My edits after AI output:**
    - Implemented plotting with pandas DataFrames and seaborn heatmaps.
    - Defined explicit output directory structure (outputs/plots/).
    - Ensured plots are generated deterministically from CSV inputs (no randomness).
- **Impact / outcome:**
    - Produced clear heatmaps showing how mean VII and lock-in rate vary across sweep parameters (top_k, alpha), suitable for insertion into the final report.
- **Attribution statement:**
    - GAI was used to plan plotting approach; the final tool choice (pandas + matplotlib + seaborn) and implementation were decided, written, and validated by me.

---

### Entry ID: GAI-2026-02-08-001
- **Feature / change title:** Seeded video corpus generator — scaffolding via Antigravity Agents
- **Branch:** ai/antigravity (remote), selectively cherry-picked to main
- **PR / Commit(s):** 8fe346b, b56f2b9
- **Files touched:** src/fyp_sim/corpus/generator.py, src/fyp_sim/corpus/loader.py, src/fyp_sim/corpus/__init__.py, configs/experiment_generated.json, src/scripts/run_batch.py, src/scripts/run_sweep.py, src/fyp_sim/models.py, tests/test_corpus_generator.py, tests/test_corpus_source_switch.py
- **Goal (1–2 lines):** Replace hard-wired video objects with a parameter-driven, seed-reproducible corpus generator, making it straightforward to vary corpus size and content without touching simulation logic.
- **GAI used:** Antigravity Agents (one-off use on the ai/antigravity branch; not a repeated workflow).
- **Constraints I gave the agent:**
    - Corpus generation must be deterministic given the same seed.
    - Do not change the simulation engine or metrics.
    - Output must be compatible with the existing loader interface.
- **Prompts/instructions (summary):**
    - Instructed the agent to scaffold a seeded video corpus generator that could replace the hard-wired video list.
    - Asked it to include a basic test covering deterministic reproducibility.
- **Agent output summary (what it produced):**
    - Scaffolded `generator.py` with a seeded generator function and an initial test in `test_corpus_generator.py`.
    - Output required correction: deterministic equality checks were not correctly enforced in the first version, and there were lint issues.
- **My critical evaluation (what I accepted/rejected and why):**
    - Accepted: the core design of a seeded generator returning a reproducible list of video objects, and the overall test structure.
    - Fixed before merging: enforced proper deterministic equality checks (the initial assertions were too loose) and resolved lint errors. This is noted directly in the commit message for 8fe346b: "Enforced deterministic equality checks and fixed lint."
    - The agent's output was a partial scaffold that needed significant review and correction before it was suitable for integration. I treated it as a starting point, not a deliverable.
    - Added independently: loader wiring (`loader.py`, `__init__.py`), runner updates to support generated corpora in `run_batch.py` and `run_sweep.py`, the generated experiment config (`configs/experiment_generated.json`), and the source-switch test suite (`test_corpus_source_switch.py`). These components were not part of the agent output.
- **Verification / checks:**
    - `pytest -q` covering both `test_corpus_generator.py` and `test_corpus_source_switch.py`.
    - `pre-commit run --all-files` (lint and formatting).
    - Ran the generated corpus end-to-end via `python -m scripts.run_batch` with the new config to confirm it produced expected output.
- **My edits after AI output:**
    - Fixed deterministic equality assertions in the test.
    - Resolved all lint and formatting issues.
    - Wrote and integrated the loader, runner wiring, and experiment config independently.
    - Cherry-picked only the reviewed, corrected components from ai/antigravity into main (commits 8fe346b and b56f2b9).
- **Impact / outcome:**
    - Removed the hard-wired video object list from the simulation. Corpora are now generated from parameters, which simplifies evaluation setup and improves reproducibility across experiment configurations.
- **Attribution statement:**
    - Antigravity Agents scaffolded the generator and an initial test on the ai/antigravity branch. I corrected the output, added all integration components independently, verified correctness with tests and lint, and cherry-picked only the reviewed and corrected parts into main.

---

### Entry ID: GAI-2026-PROCESS-001
- **Feature / change title:** ChatGPT — library existence and documentation consultation (process-level)
- **Branch:** N/A (process-level; not tied to a single commit)
- **PR / Commit(s):** N/A
- **Files touched:** N/A (no code was copied directly; see attribution statement)
- **Goal (1–2 lines):** Determine whether an appropriate Python library already existed for a feature I wanted to implement, identify the relevant parts of the documentation, and get a generic usage example to orient myself before writing the actual code.
- **GAI used:** ChatGPT (process-level consultation; approximate period: development phase).
- **Prompts/instructions (summary):**
    - Asked whether a Python library existed that could support the feature I had in mind.
    - Asked which parts of the library's documentation were most relevant to my use case.
    - Asked for a short, generic usage example unrelated to this project's codebase, to understand the API surface.
- **Agent output summary (what it produced):**
    - Confirmed the library existed and pointed to the relevant documentation sections.
    - Gave a generic, standalone usage example.
- **My critical evaluation (what I accepted/rejected and why):**
    - The example was generic and did not match the project's data structures or requirements, so it could not be used directly.
    - I read the library's official documentation myself and made my own judgement about suitability.
    - The final implementation was written from scratch, adapting the concepts to the project's specific needs.
- **Verification / checks:**
    - Read the library's official documentation before implementing.
    - Verified the implementation via the standard test and lint pipeline.
- **My edits after AI output:**
    - Did not copy the example into the codebase.
    - Implemented the feature independently, using the official documentation as the primary reference.
- **Impact / outcome:**
    - Saved time on initial library discovery. The implementation itself was written independently.
- **Attribution statement:**
    - ChatGPT was used to identify that a suitable library existed and to locate the relevant documentation section. No code generated by ChatGPT was used in the project. Final implementation and all suitability decisions were made by me.

---

### Entry ID: GAI-2026-PROCESS-002
- **Feature / change title:** ChatGPT — grammar and paragraph-strength feedback on dissertation text (process-level)
- **Branch:** N/A (process-level; dissertation document is not in this repository)
- **PR / Commit(s):** N/A
- **Files touched:** N/A
- **Goal (1–2 lines):** Catch easy-to-miss grammar issues and get high-level feedback on whether individual paragraphs were clearly structured, before finalising sections of the dissertation.
- **GAI used:** ChatGPT (process-level; approximate period: dissertation writing phase).
- **Prompts/instructions (summary):**
    - Pasted individual paragraphs and asked for grammar corrections and feedback on paragraph strength and clarity.
    - Did not ask ChatGPT to rewrite or rephrase text for submission.
- **Agent output summary (what it produced):**
    - Flagged grammar issues and suggested structural improvements at the paragraph level.
    - In some cases offered alternative phrasing.
- **My critical evaluation (what I accepted/rejected and why):**
    - Grammar corrections were reviewed individually in context; not all were accepted.
    - Structural suggestions were treated as feedback to consider, not instructions to follow.
    - Suggested alternative phrasing was not used directly; any rewrites were done manually in my own words.
- **Verification / checks:**
    - All final wording was written and revised by me.
    - Read back each revised section to confirm the voice and reasoning remained my own.
- **My edits after AI output:**
    - Applied only grammar fixes I agreed with after reading in context.
    - Rewrote any structurally weak paragraphs myself rather than using suggested alternatives.
- **Impact / outcome:**
    - Helped catch a small number of grammar issues and prompted me to revisit a few paragraphs that were unclear. Final dissertation text was written and revised by me throughout.
- **Attribution statement:**
    - ChatGPT was used for proofreading-style grammar feedback and high-level structural suggestions only. No ChatGPT-generated text appears in the submitted dissertation. All writing is my own.

---

### Entry ID: GAI-2026-04-19-001
- **Feature / change title:** Windows compatibility fixes — artefact and plotting workflow
- **Branch:** feat/windows-debug-pass
- **PR / Commit(s):** 13f0c5a
- **Files touched:** src/fyp_sim/artefacts.py, src/scripts/make_plots.py, tests/test_plotting_generation.py, tests/test_run_local_form.py, ui/run_inspector.py, .gitignore
- **Goal (1–2 lines):** Diagnose and resolve a small set of Windows-specific compatibility issues in the artefact handling and plotting workflow after switching development environment from Mac to Windows.
- **GAI used:** Claude Code (used on a dedicated branch to assist with diagnosis and fix suggestions).
- **Prompts/instructions (summary):**
    - Described the Windows-specific failures encountered in the artefact and plotting pipeline.
    - Asked Claude Code to identify the likely cause and suggest targeted fixes for path handling and test compatibility.
- **Agent output summary (what it produced):**
    - Identified Windows path separator and related compatibility issues across the affected files.
    - Suggested specific small fixes to artefacts.py, make_plots.py, run_inspector.py, and the affected tests.
- **My critical evaluation (what I accepted/rejected and why):**
    - Reviewed each proposed change individually.
    - Kept only the fixes that were clearly necessary for Windows compatibility.
    - Rejected anything that went beyond the immediate compatibility issue or introduced broader changes.
- **Verification / checks:**
    - `pytest -q` on Windows after applying fixes to confirm tests passed.
    - `pre-commit run --all-files`.
    - Confirmed that artefact generation and plotting ran without errors on Windows.
- **My edits after AI output:**
    - Reviewed all changes before accepting them.
    - Only kept the minimal set of compatibility fixes; did not accept broader refactors or unrelated suggestions.
- **Impact / outcome:**
    - Resolved Windows-specific failures in the artefact and plotting workflow. Six files were touched; scope was limited to compatibility adjustments.
- **Attribution statement:**
    - Claude Code was used to diagnose compatibility issues and suggest fixes on a dedicated branch. I reviewed each proposed change, kept only what was necessary, and verified the result with tests before merging.

---

## Notes for assessors
- GAI-assisted code work was carried out on dedicated branches (`ai/antigravity`, `feat/windows-debug-pass`) and merged into `main` after review.
- All GAI-assisted code contributions were verified with tests and lint, and reviewed for correctness, reproducibility, and scope before merging.
- Process-level GAI use (library consultation, grammar feedback) produced no code or text that appears directly in the project or dissertation.
