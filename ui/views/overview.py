# ui/views/overview.py
from __future__ import annotations

import streamlit as st

from ui.views.overview_content import (
    ARTEFACT_TREE,
    DASHBOARD_CAPABILITIES,
    PARAMETER_GROUPS,
    PRESET_GUIDE,
    PROJECT_SUMMARY,
    RESULT_GUIDE,
)


def _render_page_link(page: str, *, label: str, icon: str, help_text: str) -> None:
    if hasattr(st, "page_link"):
        st.page_link(page, label=label, icon=icon, help=help_text)
        return

    st.write(f"**{label}**")
    st.caption(f"Open `{page}` from the sidebar. {help_text}")


def _render_bullets(items: list[str]) -> None:
    st.markdown("\n".join(f"- {item}" for item in items))


def _render_eval_card(config: dict[str, str]) -> None:
    with st.container(border=True):
        st.markdown(f"**{config['name']}**")
        st.markdown(f"- **What it tests:** {config['tests']}")
        st.markdown(f"- **What to look for:** {config['look_for']}")
        st.markdown(f"- **Expected interpretation:** {config['expected']}")


def _render_parameter_groups() -> None:
    for group in PARAMETER_GROUPS:
        with st.expander(str(group["title"]), expanded=False):
            items = group["items"]
            st.markdown("\n".join(f"- **`{name}`** - {description}" for name, description in items))


def _render_result_guide() -> None:
    for title, body in RESULT_GUIDE:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)


def render() -> None:
    st.header("Overview")
    st.caption(
        "Examiner-facing guide to the dissertation evaluation workflow, "
        "run artefacts, and result interpretation."
    )

    st.info(
        "Recommended journey: create a run in Run Locally, inspect it in "
        "Explore Results, generate plots, review the artefacts, and then use "
        "Compare Runs once you have at least two completed runs."
    )
    st.caption("Use the left sidebar to move between pages.")

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        _render_page_link(
            "pages/2_Run_Locally.py",
            label="Open Run Locally",
            icon="▶️",
            help_text="Create a new local heuristic run.",
        )
    with nav2:
        _render_page_link(
            "pages/3_Explore_Results.py",
            label="Open Explore Results",
            icon="📂",
            help_text="Inspect generated run artefacts and figures.",
        )
    with nav3:
        _render_page_link(
            "pages/5_Compare_Runs.py",
            label="Open Compare Runs",
            icon="📊",
            help_text="Compare two completed runs side-by-side.",
        )

    tabs = st.tabs(
        [
            "Start here",
            "Evaluation presets",
            "Parameters",
            "Interpreting results",
            "Artefacts and scope",
        ]
    )

    with tabs[0]:
        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Project summary")
            _render_bullets(PROJECT_SUMMARY)

            st.subheader("What the dashboard lets the examiner do")
            _render_bullets(DASHBOARD_CAPABILITIES)

        with right:
            st.subheader("Recommended examiner journey")
            st.markdown(
                """
1. Open **Run Locally**.
2. Choose a preset scenario, adjust bounded parameters, and optionally add advanced JSON overrides.
3. Run the simulation and wait for the run directory to be created.
4. Open **Explore Results** and select the generated run.
5. Use the **Plots** tab to generate or review figures.
6. Review the **Overview**, **Summary**, **Seeds**, and **Files** tabs to inspect the evidence behind the figures.
7. Use **Compare Runs** after generating at least two runs.
                """
            )

            st.subheader("High-level simulation flow")
            st.markdown(
                """
1. **Configure** - choose a scenario and set the parameters for the run.
2. **Run** - execute the local heuristic simulation.
3. **Write artefacts** - the run is saved as a dated output directory with config, metadata, summaries, logs, and plots.
4. **Inspect and generate figures** - use Explore Results to review artefacts and create figures.
5. **Interpret** - assess convergence, lock-in, and variability against the dissertation's evaluation aims.
                """
            )

    with tabs[1]:
        st.subheader("Dissertation-aligned evaluation guide")
        st.markdown(
            "These six configurations correspond to the core evaluation conditions "
            "used to examine baseline convergence, phenotype variation, exploration "
            "sensitivity, and sentiment filtering."
        )
        for config in PRESET_GUIDE:
            _render_eval_card(config)

    with tabs[2]:
        st.subheader("Parameter guide")
        st.markdown(
            "Use this section when adjusting bounded controls or adding custom JSON. "
            "The goal is not to memorise every key, but to understand which settings "
            "change trajectory length, ranking pressure, behavioural response, and reproducibility."
        )
        _render_parameter_groups()

    with tabs[3]:
        st.subheader("How to interpret the results")
        st.markdown(
            "Read the figures as part of a structured evidence chain: resolved config, "
            "manifest, summary, seed logs, and then plots."
        )
        _render_result_guide()

    with tabs[4]:
        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Where results and figures live")
            st.markdown(
                "Completed runs are stored under `outputs/runs/` in a dated run "
                "directory. The main evidence path is: generate a run, select it in "
                "Explore Results, inspect the artefacts, and then generate or review "
                "the figures for that same run."
            )
            st.code(ARTEFACT_TREE, language="text")
            st.markdown(
                """
- **`config_resolved.json`** - the exact resolved configuration snapshot used for reproducibility.
- **`manifest.json`** - run metadata and high-level context.
- **`summary.csv`** - aggregated outputs used for quick interpretation.
- **`plots/`** - generated figures for the selected run.
- **`run_log.csv`** - per-step evidence for an individual seed.
                """
            )

        with right:
            st.subheader("Limits and scope")
            st.markdown(
                """
- The examiner-facing dashboard is centred on the **deterministic heuristic workflow**.
- **LLM experiments** belong to the dissertation's comparative analysis, but are not the main examiner-run path in this dashboard.
- This is a **mechanism-focused simulation**, not a claim that the system reproduces any commercial platform in full.
- Results should be interpreted as evidence under the model's stated assumptions, not as direct proof of real-world societal causality.
- The workflow stays bounded through presets, validation, deterministic seeds, and structured run artefacts.
                """
            )
