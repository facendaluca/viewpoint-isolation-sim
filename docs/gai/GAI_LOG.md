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
- **Branch:** feat/windows-debug-pass (local only — see correction note at the end of this entry)
- **PR / Commit(s):** not resolvable in this repository (see correction note).
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
- **Correction note (added 2026-07-08):**
    - While auditing this log I found that the commit hash originally recorded here (13f0c5a) does not resolve in the repository, and the branch is not on origin. My Mac was factory reset in July 2026 and any local branches that had not been pushed were lost with it, so I can no longer point at the exact commit for this work. Rather than quietly swapping in a different hash, I am recording the original reference as unverifiable and leaving the description of the work as written. Checked against `git cat-file` and `git branch -a` on 2026-07-08.

---

### Entry ID: GAI-2026-07-08-001
- **Feature / change title:** Code scan and execution bug fixes (PR #1)
- **Branch:** fix/code-scan-bug-fixes (pushed to origin)
- **PR / Commit(s):** PR #1, merged as dc2e7f5 (my review commit on the branch: ad4e176)
- **Files touched:** src/fyp_sim/plotting/compare.py (restored), src/fyp_sim/plotting/multi_agent_plots.py, src/fyp_sim/simulation/engine.py, src/scripts/run_sweep.py, src/scripts/run_compare.py, src/fyp_sim/corpus/generator.py, src/fyp_sim/corpus/loader.py, src/fyp_sim/agents/deciders.py, pyproject.toml, tests/test_engine_viewpoint_drift.py (new), tests/test_multi_agent_plots_cohort_sizes.py (new), results/sweep_summary.csv (deleted) — 19 files in total.
- **Goal (1–2 lines):** Fix the bugs an agent code scan found that actually broke execution: the E6 two-agent plot crash, a missing compare-plot module that run_compare imported, and a run_sweep crash on a method that does not exist.
- **GAI used:** Claude Code. The scan and the fixes were produced in a separate copy of the repo, not in this one. The agent committed nothing here.
- **Constraints I gave the agent:**
    - Work in the copy, never in my repo directly.
    - Deliver the changes to my repo as a patch on a new branch, uncommitted, so I could review the diff myself before anything lands.
    - Fix bugs only; no behaviour changes beyond what the scan flagged.
- **Prompts/instructions (summary):**
    - Asked for a full scan of the project for execution-breaking and correctness bugs, with a written report.
    - Then asked for the bug-fix commit to be applied to my repo on a review branch without committing.
- **Agent output summary (what it produced):**
    - Restored `plotting/compare.py`, which `run_compare` imported but which had been deleted.
    - Fixed the multi-agent lock-in plot assuming exactly three phenotypes, which crashed E6 (two agents), and added a regression test covering two-, three-, and four-agent cohorts.
    - Fixed `run_sweep` calling `User.clone()`, which does not exist.
    - Made the engine resolve `drift_alpha` consistently in both drift branches, with a regression test.
    - `run_compare` now passes `enable_viewpoint_drift` through and builds a fresh user per (agent, seed) so state cannot leak between runs.
    - Reworded some AI-sounding comments in the corpus module and deleted a stale `results/sweep_summary.csv` that used the old alpha schema.
- **My critical evaluation (what I accepted/rejected and why):**
    - Reviewed the whole patch as a working-tree diff before committing anything.
    - Accepted all of it: each fix addressed a failure I could reproduce (the E6 plot crash, the sweep crash) or came with a regression test making the behaviour explicit.
- **Verification / checks:**
    - `pytest -q` (164 tests at that point, all passing).
    - `ruff check .`
    - Ran the dashboard and regenerated the E6 figures to confirm the crash was gone.
- **My edits after AI output:**
    - Committed the reviewed patch myself as ad4e176 and merged it through PR #1.
- **Impact / outcome:**
    - The comparison workflow and the E6 figures work again, and the sweep no longer crashes.
- **Attribution statement:**
    - Claude Code found the bugs and wrote the fixes in a copy of the repo. I reviewed the patch line by line in my own repo, committed it myself, and merged it. The reviewed patch is kept at `docs/gai/diffs/pr1-bug-fixes.patch`.

---

### Entry ID: GAI-2026-07-08-002
- **Feature / change title:** Dissertation alignment of eval configs and metric definitions (PR #2)
- **Branch:** fix/dissertation-alignment (pushed to origin)
- **PR / Commit(s):** PR #2, merged as 93b744c (my review commit on the branch: 6a9e34b); my own follow-up refactor d3cfc91.
- **Files touched:** configs/eval/E1–E6 (all six), src/fyp_sim/engagement.py, src/fyp_sim/plotting/common.py, src/fyp_sim/plotting/multi_agent_metrics.py, src/fyp_sim/plotting/multi_run_metrics.py, src/fyp_sim/simulation/engine.py, src/fyp_sim/simulation/viewpoint_drift.py, tests/test_engagement.py, tests/test_multi_agent_metrics.py — 14 files.
- **Goal (1–2 lines):** Bring the implementation in line with what Chapter 3 actually says: Table 3.9 corpus topic weights in the E1–E6 configs, watch time driven by the decided action, time-to-lock-in reported as the persistence-window-confirmed step, and Student's t confidence intervals at n=5 seeds.
- **GAI used:** Claude Code, same patch-and-review workflow as PR #1. The same session also fixed the type-checker errors that appeared in `plotting/common.py` after my environment rebuild pulled in pandas 3.
- **Constraints I gave the agent:**
    - Same as PR #1: patch on a review branch, nothing committed by the agent.
    - Any comments it writes have to read like normal Python comments, not AI text.
- **Prompts/instructions (summary):**
    - Asked for the alignment commit from the repo copy to be applied here on its own branch, uncommitted.
    - Separately asked it to fix the errors showing in `common.py` and to explain what caused them.
- **Agent output summary (what it produced):**
    - E1–E6 corpus topic weights changed to the Table 3.9 ratios (comedy 0.30 / hobbies 0.50 / politics 0.10 / other 0.10).
    - Watch-time mapping now follows the decider's actual action, with a regression test — matters most for LLM mode, where the logged action could previously disagree with the watch-time signal.
    - Time-to-lock-in unified as the window-confirmed step across summary.csv and the plots.
    - CI bands use the t multiplier instead of 1.96 at small seed counts.
    - Corrected the viewpoint-drift docstring (the code was right; the docstring said 0.5 where the pseudocode says 0.2).
    - Fixed 31 type-checker errors in `common.py`: two annotation bugs that predated the port and a batch caused by pandas 3 shipping stricter type stubs. No behaviour change — verified by the tests.
- **My critical evaluation (what I accepted/rejected and why):**
    - Checked the new weight ratios against Table 3.9 myself before committing.
    - Accepted the metric redefinitions because they match what the methodology chapter describes; the old episode-start timing was the thing out of line, not the dissertation.
    - Made a few adjustments of my own before committing, and did a small follow-up refactor (d3cfc91) afterwards.
- **Verification / checks:**
    - `pytest -q` (165 tests, including the new engagement regression test).
    - `ruff check .`
    - Full E1–E6 rerun with regenerated plots afterwards, since these changes deliberately alter results — old outputs are superseded and are not to be used in Chapter 5.
- **My edits after AI output:**
    - Committed as 6a9e34b with my adjustments, merged through PR #2, then tidied types further in d3cfc91.
- **Impact / outcome:**
    - The experiments now measure what the dissertation says they measure. E1–E6 evidence regenerated from the aligned configs.
- **Attribution statement:**
    - Claude Code produced the alignment changes and the type fixes in patch form. I reviewed the diff, adjusted parts of it, committed and merged it myself. The reviewed patch is kept at `docs/gai/diffs/pr2-dissertation-alignment.patch`.

---

### Entry ID: GAI-2026-PROCESS-003
- **Feature / change title:** Claude Code — post-merge verification runs and examiner-style code scan (process-level)
- **Branch:** N/A (no source changes; evidence artefacts only)
- **PR / Commit(s):** N/A
- **Files touched:** CODE_SCAN_MARK_RISK_SUMMARY.md (generated report), run artefacts under outputs/
- **Goal (1–2 lines):** After merging PRs #1 and #2, confirm that main actually runs every documented workflow, and get a strict mark-risk review of the repo before Chapter 5 is written from the new outputs.
- **GAI used:** Claude Code (2026-07-08), run on my machines under my supervision.
- **Prompts/instructions (summary):**
    - Asked it to check main runs all experiments with no bugs, then to rerun everything the port-review verdict document called for.
    - Asked for a full examiner-style scan reporting only issues that could realistically cost marks, with evidence for every finding.
- **Agent output summary (what it produced):**
    - Reran E1–E6 with plot regeneration (summaries byte-identical across two independent reruns, which is good reproducibility evidence), the parameter sweep, and the heuristic-vs-LLM compare.
    - Wrote CODE_SCAN_MARK_RISK_SUMMARY.md with severity-ranked findings; it found the dead reference in this log, which the correction note above addresses.
- **My critical evaluation (what I accepted/rejected and why):**
    - One evidence decision to record: the compare config named `llama-3.2-3b-instruct`, which is no longer available on my LLM machine. The compare evidence run (`outputs/compare/compare__62c15b1d3b`) therefore uses `qwen2.5-1.5b-instruct` — the smallest model available that returned valid decision-contract JSON (the 3B and 12B models failed the JSON contract on every call and only ever produced heuristic fallbacks). The dissertation's config listings need updating to match before submission.
    - The scan findings are the agent's opinion, not fixes; I decide which ones to act on.
- **Verification / checks:**
    - All runs executed through the project venv; commands and pass/fail results are listed in CODE_SCAN_MARK_RISK_SUMMARY.md.
- **Attribution statement:**
    - Claude Code executed the verification runs and wrote the scan report. No dissertation text and no simulation code came from this session; the run artefacts are outputs of my own experiment configs.

---

## Notes for assessors
- GAI-assisted code work was carried out on dedicated branches (`ai/antigravity`, `feat/windows-debug-pass`, `fix/code-scan-bug-fixes`, `fix/dissertation-alignment`) and merged into `main` after review.
- For the July 2026 work, the agent operated in a separate copy of the repository; changes reached this repository only as patches on review branches, which I reviewed as uncommitted diffs, committed, and merged myself (PRs #1 and #2). The reviewed patches are kept under `docs/gai/diffs/`.
- All GAI-assisted code contributions were verified with tests and lint, and reviewed for correctness, reproducibility, and scope before merging.
- Process-level GAI use (library consultation, grammar feedback) produced no code or text that appears directly in the project or dissertation.
