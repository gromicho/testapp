# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import re
import ast

# ---------------------------------------------------------------------
# Hardcoded grid
# ---------------------------------------------------------------------
GRID = np.array([
    [14, 4, 0, 0, 11, 0, 0, 0, 0, 0, 4, 0, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 4, 7, 0, 11, 4, 0, 0, 0],
    [0, 0, 4, 11, 7, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 7, 7, 7, 14, 14, 4, 0, 0],
    [0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 4, 11, 4, 0, 4, 0, 4, 4],
    [0, 0, 0, 4, 4, 7, 7, 11, 7, 0, 0, 0, 4, 0, 7, 0, 0, 0, 4, 0, 11, 4, 0, 4, 0, 0, 4, 0, 0, 0],
    [0, 0, 4, 11, 4, 4, 0, 4, 11, 4, 7, 4, 4, 0, 7, 4, 11, 7, 4, 4, 11, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 7, 18, 7, 4, 0, 7, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 4, 4, 0, 4, 7, 7, 0, 4, 14, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 0],
    [0, 7, 0, 4, 0, 0, 4, 0, 7, 4, 11, 7, 11, 14, 14, 14, 4, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 11, 0],
    [0, 4, 0, 0, 4, 0, 0, 4, 7, 0, 0, 0, 18, 4, 4, 0, 0, 7, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 4, 4],
    [7, 0, 7, 4, 0, 0, 7, 7, 0, 0, 7, 4, 0, 7, 4, 7, 4, 7, 4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 4],
    [0, 4, 0, 11, 0, 4, 0, 4, 0, 11, 4, 4, 14, 4, 7, 25, 7, 0, 4, 0, 0, 0, 4, 4, 0, 0, 4, 4, 4, 4],
    [0, 11, 4, 4, 0, 0, 0, 0, 7, 18, 0, 4, 0, 7, 14, 4, 11, 4, 0, 4, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 7, 4, 4, 4, 0, 4, 4, 0, 4, 4, 11, 0, 0, 0, 4, 4, 0, 7, 0, 0, 0, 0, 0, 4, 0, 0],
    [7, 4, 0, 4, 4, 0, 0, 11, 7, 4, 0, 18, 4, 0, 14, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 0],
    [0, 0, 0, 0, 4, 4, 4, 0, 4, 0, 0, 0, 11, 4, 0, 0, 0, 0, 7, 4, 11, 14, 7, 0, 0, 7, 0, 4, 4, 0],
    [0, 0, 0, 4, 4, 4, 7, 0, 0, 11, 7, 0, 4, 11, 22, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0],
    [0, 0, 4, 11, 0, 0, 0, 4, 0, 7, 4, 0, 0, 4, 0, 0, 0, 4, 14, 7, 4, 4, 4, 0, 0, 4, 0, 7, 7, 4],
    [4, 0, 0, 0, 4, 0, 4, 0, 14, 14, 7, 4, 4, 0, 7, 4, 4, 0, 4, 0, 0, 0, 14, 4, 18, 7, 4, 0, 0, 0],
    [0, 7, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 4, 11, 0, 0, 4, 0, 7, 7, 11, 0, 4, 0, 4, 4, 0, 0, 0, 4],
    [4, 4, 0, 4, 0, 0, 0, 4, 4, 7, 4, 4, 0, 0, 0, 0, 7, 0, 0, 0, 11, 0, 4, 4, 4, 0, 0, 0, 0, 0]
])

# ---------------------------------------------------------------------
# Direction data
# ---------------------------------------------------------------------
DIR_VECTORS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
STEP_LENGTH = [5, 7, 5, 7, 5, 7, 5, 7]
DIR_TO_IDX = {"N": 0, "NE": 1, "E": 2, "SE": 3,
              "S": 4, "SW": 5, "W": 6, "NW": 7}
IDX_TO_DIR = {v: k for k, v in DIR_TO_IDX.items()}


# ---------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------
def parse_day_paths(text):
    s = text.strip()
    if not s:
        raise ValueError("Empty input.")
    try:
        normalized = s.replace("(", "[").replace(")", "]")
        obj = ast.literal_eval(normalized)
        if isinstance(obj, tuple):
            obj = [list(obj)]
        if isinstance(obj, list) and all(isinstance(el, (list, tuple)) for el in obj):
            if all(isinstance(x, (int, float)) for x in obj[0]) and len(obj[0]) == 2:
                obj = [obj]
            parsed = [[tuple(map(int, step)) for step in day] for day in obj]
            summary = _summarize_paths(parsed)
            return parsed, summary
    except Exception:
        pass
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    all_days = []
    coord_pattern = re.compile(r"(-?\d+)\s*[,; ]\s*(-?\d+)")
    for ln in lines:
        coords = coord_pattern.findall(ln)
        if coords:
            all_days.append([tuple(map(int, c)) for c in coords])
    if not all_days:
        raise ValueError("Could not interpret any coordinate pairs.")
    summary = _summarize_paths(all_days)
    return all_days, summary


def _summarize_paths(day_paths):
    n_days = len(day_paths)
    n_coords = sum(len(day) for day in day_paths)
    summary_lines = [f"🧩 Parsed {n_days} day{'s' if n_days > 1 else ''}, {n_coords} coordinates total."]
    for i, day in enumerate(day_paths, start=1):
        summary_lines.append(f"• Day {i}: {day}")
    return "\n".join(summary_lines)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_day_paths(grid, day_paths, start_cell, start_d, max_days, max_distance_per_day):
    rows, cols = grid.shape
    visited = set()
    plastic_by_day = []
    distance_by_day = []
    distance_by_day_steps = []
    prev_end = start_cell
    prev_dir_idx = DIR_TO_IDX[start_d]
    y_start, x_start = start_cell
    if not (0 <= y_start < rows and 0 <= x_start < cols):
        return False, f"Start cell {start_cell} out of bounds", [], [], []
    visited.add(start_cell)
    if len(day_paths) > max_days:
        return False, f"Too many days ({len(day_paths)} > {max_days})", [], [], []

    for d, day in enumerate(day_paths, start=1):
        if not day or len(day) < 2:
            return False, f"Day {d} is empty or too short", [], [], []
        y0, x0 = day[0]
        if d == 1 and (y0, x0) != start_cell:
            return False, f"Day 1 must start at {start_cell}, got {(y0, x0)}", [], [], []
        if d > 1 and (y0, x0) != prev_end:
            return False, f"Day {d} starts at {(y0, x0)} but previous ended at {prev_end}", [], [], []

        dist = 0
        steps = []
        plastic_today = []
        if (y0, x0) not in visited:
            plastic_today.append(int(grid[y0, x0]))
            visited.add((y0, x0))
        last_dir_idx = prev_dir_idx

        for i in range(1, len(day)):
            y1, x1 = day[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                return False, f"Day {d} step {i} out of bounds: {(y1, x1)}", [], [], []
            dy, dx = y1 - y0, x1 - x0
            try:
                dir_idx = DIR_VECTORS.index((dy, dx))
            except ValueError:
                return False, f"Day {d} step {i} invalid move from {(y0, x0)} to {(y1, x1)}", [], [], []
            diff = abs(dir_idx - last_dir_idx)
            diff = min(diff, 8 - diff)
            if diff > 1:
                return False, (
                    f"Day {d} step {i} turns too sharply: {IDX_TO_DIR[last_dir_idx]} → {IDX_TO_DIR[dir_idx]}"
                ), [], [], []
            step_len = STEP_LENGTH[dir_idx]
            if dist + step_len > max_distance_per_day:
                return False, f"Day {d} exceeds distance {max_distance_per_day} at step {i}", [], [], []
            dist += step_len
            steps.append(step_len)
            if (y1, x1) not in visited:
                plastic_today.append(int(grid[y1, x1]))
                visited.add((y1, x1))
            y0, x0 = y1, x1
            last_dir_idx = dir_idx

        prev_end = (y0, x0)
        prev_dir_idx = last_dir_idx
        plastic_by_day.append(plastic_today)
        distance_by_day.append(dist)
        distance_by_day_steps.append(steps)

    return True, "Path validated successfully", plastic_by_day, distance_by_day, distance_by_day_steps


# ---------------------------------------------------------------------
# Draw final frame
# ---------------------------------------------------------------------
def draw_last_frame(grid, day_paths, plastic_by_day, distance_by_day_steps, fig_width, fig_height):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(grid, ax=ax, cmap="YlGnBu", annot=True, fmt="d", cbar=False, square=True)
    ax.invert_yaxis()

    cmap = plt.get_cmap("tab10")
    day_color_map = {i: cmap(i % 10) for i in range(len(day_paths))}
    visited = set()
    move_counter = 0

    for day_index, day in enumerate(day_paths):
        color = day_color_map[day_index]
        for j in range(1, len(day)):
            (y0, x0), (y1, x1) = day[j - 1], day[j]
            x0c, y0c = x0 + 0.5, y0 + 0.5
            x1c, y1c = x1 + 0.5, y1 + 0.5
            xm, ym = (x0c + x1c) / 2, (y0c + y1c) / 2
            ax.annotate("", xy=(xm, ym), xytext=(x0c, y0c),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=2)
            move_counter += 1
            ax.text(x0c, y0c + 0.25, str(move_counter),
                    color="black", fontsize=8, ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9))
            rect = patches.FancyBboxPatch(
                (x0, y0), 1, 1,
                boxstyle="round,pad=0.002,rounding_size=0.15",
                linewidth=3, edgecolor=color, facecolor="none", alpha=0.8)
            ax.add_patch(rect)

    y_start, x_start = day_paths[0][0]
    ax.add_patch(patches.FancyBboxPatch(
        (x_start, y_start), 1, 1, boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=3, edgecolor="green", facecolor="none", alpha=0.8))
    last_day_index = len(day_paths) - 1
    last_color = day_color_map[last_day_index]
    last_y, last_x = day_paths[last_day_index][-1]
    ax.add_patch(patches.FancyBboxPatch(
        (last_x, last_y), 1, 1, boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=9, edgecolor=last_color, facecolor="none", alpha=0.8))

    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in distance_by_day_steps)
    ax.set_title(f"plastic = {plastic_total}    |    distance = {distance_total}",
                 fontsize=13, family="monospace", pad=15)
    legend_handles = [patches.Patch(color=day_color_map[i], label=f"Day {i + 1}")
                      for i in range(len(day_paths))]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

    buf = BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    pdf_bytes = buf.read()
    buf.close()
    return fig, pdf_bytes


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("Validate and Plot Grid Path (with Start Cell + Direction)")

example = '0,0  0,1  0,2\n0,2  0,3  0,4'
path_str = st.text_area("Enter day_paths (flexible format):", example)
start_y = st.number_input("Start row (y):", min_value=0, max_value=GRID.shape[0] - 1, value=0)
start_x = st.number_input("Start column (x):", min_value=0, max_value=GRID.shape[1] - 1, value=0)
start_dir = st.selectbox("Start direction:", list(DIR_TO_IDX.keys()), index=2)
max_days = st.number_input("Max days:", min_value=1, max_value=10, value=5)
max_distance = st.number_input("Max distance per day:", min_value=5, max_value=50, value=20)

if st.button("Validate and Draw"):
    try:
        day_paths, summary = parse_day_paths(path_str)
        st.info(summary)
    except Exception as e:
        st.error(f"⚠️ Parsing error: {e}")
        st.stop()

    ok, msg, plastic_by_day, distance_by_day, distance_by_day_steps = validate_day_paths(
        GRID, day_paths,
        start_cell=(start_y, start_x),
        start_d=start_dir,
        max_days=max_days,
        max_distance_per_day=max_distance
    )

    if ok:
        st.success("✅ Path valid")
        st.info(msg)

        for d, (plastics, dist_steps) in enumerate(zip(plastic_by_day, distance_by_day_steps), start=1):
            st.write(f"**Day {d}:** Plastic = {sum(plastics)} ({'+'.join(map(str, plastics))})  |  "
                     f"Distance = {sum(dist_steps)} ({'+'.join(map(str, dist_steps))})")

        fig, pdf_bytes = draw_last_frame(GRID, day_paths, plastic_by_day, distance_by_day_steps)
        st.pyplot(fig, clear_figure=True)
        st.download_button(
            label="Download last frame as PDF",
            data=pdf_bytes,
            file_name="last_frame.pdf",
            mime="application/pdf"
        )
    else:
        st.error("❌ Invalid path")
        st.warning(msg)
