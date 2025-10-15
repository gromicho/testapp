# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# ---------------------------------------------------------------------
# Hardcoded grid
# ---------------------------------------------------------------------
GRID = np.array([
    [4, 0, 4, 7, 7, 0, 4, 14],
    [7, 11, 14, 14, 14, 4, 0, 7],
    [0, 18, 4, 4, 0, 0, 7, 4],
    [4, 0, 7, 4, 7, 4, 7, 4],
    [4, 14, 4, 7, 25, 7, 0, 4],
    [4, 0, 7, 14, 4, 11, 4, 0],
    [0, 4, 4, 11, 0, 0, 0, 4],
    [18, 4, 0, 14, 0, 0, 0, 0]
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
# Enhanced validation with messages
# ---------------------------------------------------------------------
def validate_day_paths(grid, day_paths, start_d, max_days, max_distance_per_day):
    rows, cols = grid.shape
    visited = set()
    plastic_by_day = []
    distance_by_day = []
    distance_by_day_steps = []
    current_dir = start_d
    prev_end = None
    prev_dir = start_d

    if not isinstance(day_paths, list):
        return False, "day_paths must be a list", [], [], []
    if len(day_paths) > max_days:
        return False, f"Too many days ({len(day_paths)} > {max_days})", [], [], []

    for d, day in enumerate(day_paths, start=1):
        if not day or len(day) < 2:
            return False, f"Day {d} is empty or too short", [], [], []

        dist = 0
        steps = []
        plastic_today = []

        y0, x0 = day[0]
        if not (0 <= y0 < rows and 0 <= x0 < cols):
            return False, f"Day {d} starts out of bounds at {(y0, x0)}", [], [], []

        if d > 1 and (y0, x0) != prev_end:
            return False, f"Day {d} starts at {(y0, x0)} but previous ended at {prev_end}", [], [], []

        if (y0, x0) not in visited:
            plastic_today.append(int(grid[y0, x0]))
            visited.add((y0, x0))

        for i in range(1, len(day)):
            y1, x1 = day[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                return False, f"Day {d} step {i} out of bounds: {(y1, x1)}", [], [], []

            dy, dx = y1 - y0, x1 - x0
            try:
                idx = DIR_VECTORS.index((dy, dx))
            except ValueError:
                return False, f"Day {d} step {i} invalid move from {(y0, x0)} to {(y1, x1)}", [], [], []

            step_len = STEP_LENGTH[idx]
            if dist + step_len > max_distance_per_day:
                return False, f"Day {d} exceeds distance {max_distance_per_day} at step {i}", [], [], []

            dist += step_len
            steps.append(step_len)

            if (y1, x1) not in visited:
                plastic_today.append(int(grid[y1, x1]))
                visited.add((y1, x1))

            y0, x0 = y1, x1
            current_dir = IDX_TO_DIR[idx]

        prev_end = (y0, x0)
        prev_dir = current_dir
        plastic_by_day.append(plastic_today)
        distance_by_day.append(dist)
        distance_by_day_steps.append(steps)

    return True, "Path validated successfully", plastic_by_day, distance_by_day, distance_by_day_steps


# ---------------------------------------------------------------------
# Draw final frame with totals and return both fig and PDF bytes
# ---------------------------------------------------------------------
def draw_last_frame(grid, day_paths, plastic_by_day, distance_by_day_steps):
    fig, ax = plt.subplots(figsize=(6, 6))
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

            # Arrow between cells
            ax.annotate("", xy=(xm, ym), xytext=(x0c, y0c),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=2)

            # Step number
            move_counter += 1
            ax.text(x0c, y0c + 0.25, str(move_counter),
                    color="black", fontsize=8, ha="center", va="center", weight="bold",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9))

            # Cell frame
            rect = patches.FancyBboxPatch(
                (x0, y0), 1, 1,
                boxstyle="round,pad=0.002,rounding_size=0.15",
                linewidth=3,
                edgecolor=color,
                facecolor="none",
                alpha=0.8,
                zorder=3 + day_index
            )
            ax.add_patch(rect)

            # Plastic value
            if (y0, x0) not in visited:
                visited.add((y0, x0))
                ax.text(x0 + 0.5, y0 + 0.5, str(grid[y0, x0]),
                        ha="center", va="center", fontsize=10, color=color,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  edgecolor=color, linewidth=1.5,
                                  facecolor="white", alpha=1), zorder=4)

    # Highlight start cell (green)
    y_start, x_start = day_paths[0][0]
    start_rect = patches.FancyBboxPatch(
        (x_start, y_start), 1, 1,
        boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=3, edgecolor="green",
        facecolor="none", alpha=0.8, zorder=10
    )
    ax.add_patch(start_rect)

    # Highlight final cell (thick)
    last_day_index = len(day_paths) - 1
    last_color = day_color_map[last_day_index]
    last_y, last_x = day_paths[last_day_index][-1]

    end_rect = patches.FancyBboxPatch(
        (last_x, last_y), 1, 1,
        boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=9, edgecolor=last_color,
        facecolor="none", alpha=0.8, zorder=12
    )
    ax.add_patch(end_rect)

    # Plastic and distance totals in title
    plastic_exprs = ["+".join(str(x) for x in p) for p in plastic_by_day if p]
    dist_exprs = ["+".join(str(x) for x in d) for d in distance_by_day_steps if d]
    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in distance_by_day_steps)

    plastic_line = f'{"plastic":>13}: {" + ".join(plastic_exprs)} = {plastic_total}'
    distance_line = f'{"distance":>13}: {" + ".join(dist_exprs)} = {distance_total}'
    ax.set_title(f"{plastic_line}\n{distance_line}", fontsize=12, family="monospace")

    # --- NEW: Legend showing color per day ---
    legend_handles = [
        patches.Patch(color=day_color_map[i], label=f"Day {i + 1}")
        for i in range(len(day_paths))
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

    # Export to PDF bytes
    buf = BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    pdf_bytes = buf.read()
    buf.close()

    return fig, pdf_bytes



# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("Validate and Plot Grid Path (Inline + PDF Download)")

example = '[[[0,0],[0,1],[0,2]], [[4,4],[4,5],[4,6]]]'
path_str = st.text_area("Enter day_paths (list of lists of [y,x]):", example)
start_dir = st.selectbox("Start direction:", list(DIR_TO_IDX.keys()), index=2)
max_days = st.number_input("Max days:", min_value=1, max_value=10, value=5)
max_distance = st.number_input("Max distance per day:", min_value=5, max_value=50, value=20)

if st.button("Validate and Draw"):
    try:
        day_paths = eval(path_str)
        ok, msg, plastic, dist, dist_steps = validate_day_paths(
            GRID, day_paths, start_d=start_dir,
            max_days=max_days, max_distance_per_day=max_distance
        )
        if ok:
            st.success("✅ Path valid")
            st.info(msg)
            fig, pdf_bytes = draw_last_frame(GRID, day_paths, plastic, dist_steps)
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
    except Exception as e:
        st.error(f"⚠️ Parsing error: {e}")
