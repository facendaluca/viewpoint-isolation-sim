from __future__ import annotations

from pathlib import Path

import streamlit as st

_SUPPORTED_PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def list_plot_images(plots_dir: Path) -> list[Path]:
    """Return supported plot image files under plots/"""
    if not plots_dir.is_dir():
        return []

    plot_files = [
        path
        for path in plots_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_PLOT_SUFFIXES
    ]
    return sorted(plot_files, key=lambda p: str(p.relative_to(plots_dir)))


def image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _relative_label(path: Path, plots_dir: Path) -> str:
    return str(path.relative_to(plots_dir))


def _state_key(plots_dir: Path, name: str) -> str:
    safe_dir = str(plots_dir).replace("\\", "__").replace("/", "__").replace(":", "_")
    return f"figure_browser::{safe_dir}::{name}"


def filter_plot_images(plot_files: list[Path], plots_dir: Path, query: str) -> list[Path]:
    """Filter plot files by case-insensitive filename/path match."""
    if not query.strip():
        return plot_files

    needle = query.casefold().strip()
    return [path for path in plot_files if needle in _relative_label(path, plots_dir).casefold()]


def sort_plot_images(
    plot_files: list[Path],
    plots_dir: Path,
    descending: bool = False,
) -> list[Path]:
    """Sort plot files by relative path for stable ordering."""
    return sorted(
        plot_files,
        key=lambda p: _relative_label(p, plots_dir),
        reverse=descending,
    )


def _ensure_selected_label(labels: list[str], selected_label_key) -> str:
    """Ensure a valid selected label exists in session state."""
    current = st.session_state.get(selected_label_key)
    if current not in labels:
        st.session_state[selected_label_key] = labels[0]
    return st.session_state[selected_label_key]


def _shift_selected_label(labels: list[str], selected_label_key: str, delta: int) -> None:
    """Moe the current selection forwards/backwards by delta."""
    current = _ensure_selected_label(labels, selected_label_key)
    current_index = labels.index(current)
    new_index = max(0, min(len(labels) - 1, current_index + delta))
    st.session_state[selected_label_key] = labels[new_index]


def _path_from_label(plot_files: list[Path], plots_dir: Path, label: str) -> Path:
    for path in plot_files:
        if _relative_label(path, plots_dir) == label:
            return path
    return plot_files[0]


def _render_selected_figure(
    selected_plot: Path, plots_dir: Path, position: int, total: int
) -> None:
    selected_label = _relative_label(selected_plot, plots_dir)

    st.markdown("### Selected figure")

    meta_cols = st.columns(4)
    meta_cols[0].write(f"**Figure:** {position} / {total}")
    meta_cols[1].write(f"**File:** `{selected_plot.name}`")
    meta_cols[2].write(f"**Relative path:** `{selected_label}`")
    meta_cols[3].write(f"**Size:** `{selected_plot.stat().st_size / 1024:.1f} KB`")

    st.image(str(selected_plot), caption=selected_label, use_container_width=True)

    st.download_button(
        f"Download {selected_plot.name}",
        data=selected_plot.read_bytes(),
        file_name=selected_plot.name,
        mime=image_mime_type(selected_plot),
    )


def _render_gallery(
    plot_files: list[Path],
    plots_dir: Path,
    selected_label_key: str,
    current_selected_label: str,
) -> str:
    st.markdown("### Gallery")
    st.caption("Use Open to load a figure into the detail view below.")

    column_count = st.slider("Gallery columns", min_value=2, max_value=4, value=3)
    columns = st.columns(column_count)

    selected_label = current_selected_label

    for index, path in enumerate(plot_files):
        label = _relative_label(path, plots_dir)
        is_selected = label == selected_label

        with columns[index % column_count]:
            st.image(str(path), caption=label, use_container_width=True)
            if st.button(
                "Selected" if is_selected else "Open",
                key=_state_key(plots_dir, f"open::{index}::{label}"),
                disabled=is_selected,
                use_container_width=True,
            ):
                selected_label = label
                st.session_state[selected_label_key] = label

    return selected_label


def render_figure_browser(plots_dir: Path) -> None:
    """Render a reusable figure browser for a runs plot directory."""
    st.subheader("Plots")

    if not plots_dir.is_dir():
        st.info("No plots directory found for this run.")
        return

    all_plot_files = list_plot_images(plots_dir)
    if not all_plot_files:
        st.info("plots/ exists, but it contains no supported image files.")
        st.caption("Supported: .png, .jpg, .jpeg, .webp")
        return

    control_cols = st.columns([2, 1, 1])

    with control_cols[0]:
        query = st.text_input(
            "Filter figures",
            value="",
            placeholder="Type part of a filename or relative path",
        )

    with control_cols[1]:
        sort_choice = st.selectbox("Sort", options=["Name (A-Z)", "Name (Z-A)"], index=0)

    with control_cols[2]:
        view_mode = st.selectbox("View", options=["Single figure", "Gallery"], index=0)

    filtered_plot_files = filter_plot_images(all_plot_files, plots_dir, query)
    descending = sort_choice == "Name (Z-A)"
    filtered_plot_files = sort_plot_images(filtered_plot_files, plots_dir, descending=descending)

    if not filtered_plot_files:
        st.info("No figure match the current filter.")
        return

    st.caption(f"Showing {len(filtered_plot_files)} of {len(all_plot_files)} figures(s).")

    labels = [_relative_label(path, plots_dir) for path in filtered_plot_files]
    selected_label_key = _state_key(plots_dir, "selected_label")
    selected_label = _ensure_selected_label(labels, selected_label_key)

    if view_mode == "Single figure":
        selected_index = labels.index(selected_label)

        nav_cols = st.columns([1, 1, 3])
        with nav_cols[0]:
            if st.button("Previous", disabled=selected_index == 0, use_container_width=True):
                _shift_selected_label(labels, selected_label_key, -1)
                st.rerun()

        with nav_cols[1]:
            if st.button(
                "Next", disabled=selected_index == len(labels) - 1, use_container_width=True
            ):
                _shift_selected_label(labels, selected_label_key, 1)
                st.rerun()

        with nav_cols[2]:
            st.write(f"**Figure {selected_index + 1} of {len(labels)}**")

        chosen_label = st.selectbox("Select a figure", options=labels, index=selected_index)
        if chosen_label != st.session_state[selected_label_key]:
            st.session_state[selected_label_key] = chosen_label
            selected_label = chosen_label

    else:
        selected_label = _render_gallery(
            filtered_plot_files, plots_dir, selected_label_key, selected_label
        )

    selected_plot = _path_from_label(filtered_plot_files, plots_dir, selected_label)
    selected_index = labels.index(selected_label) + 1

    st.divider()
    _render_selected_figure(selected_plot, plots_dir, position=selected_index, total=len(labels))
