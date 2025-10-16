# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
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
DIR_VECTORS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]
STEP_LENGTH = [5, 7, 5, 7, 5, 7, 5, 7]
DIR_TO_IDX = {
    "N": 0, "NE": 1, "E": 2, "SE": 3,
    "S": 4, "SW": 5, "W": 6, "NW": 7
}
IDX_TO_DIR = {v: k for k, v in DIR_TO_IDX.items()}


# ---------------------------------------------------------------------
# Parse rotation input (-1, 0, 1)
# ---------------------------------------------------------------------
def parse_rotation_paths(text):
    s = text.strip()
    if not s:
        raise ValueError("Empty input.")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    all_days = []
    for ln in lines:
        normalized = ln.replace(';', ',')
        if not normalized.startswith('['):
            normalized = '[' + normalized + ']'
        vals = ast.literal_eval(normalized)
        vals = [int(v) for v in vals]
        if not all(v in (-1, 0, 1) for v in vals):
            raise ValueError(f"Invalid value in line: {ln}")
        all_days.append(vals)
    return all_days


# ---------------------------------------------------------------------
# Convert rotations to coordinates
# ---------------------------------------------------------------------
def rotations_to_coords(start_cell, start_dir, rotations):
    y, x = start_cell
    dir_idx = DIR_TO_IDX[start_dir]
    coords = [(y, x)]
    for r in rotations:
        dir_idx = (dir_idx + r) % 8
        dy, dx = DIR_VECTORS[dir_idx]
        y, x = y + dy, x + dx
        coords.append((y, x))
    return coords, dir_idx


# ---------------------------------------------------------------------
# Validate route
# ---------------------------------------------------------------------
def validate_rotation_paths(grid, rotation_days, start_cell, start_dir, max_days, max_distance):
    rows, cols = grid.shape
    visited = set()
    plastic_by_day = []
    distance_by_day = []
    distance_by_day_steps = []

    y, x = start_cell
    if not (0 <= y < rows and 0 <= x < cols):
        return False, "Start cell out of bounds", [], [], []

    dir_idx = DIR_TO_IDX[start_dir]
    visited.add(start_cell)
    prev_end = start_cell
    prev_dir = dir_idx

    for d, rotations in enumerate(rotation_days, start=1):
        if not rotations:
            return False, f"Day {d} empty.", [], [], []
        coords, new_dir = rotations_to_coords(prev_end, IDX_TO_DIR[prev_dir], rotations)

        dist = 0
        plastics = []
        steps = []

        y0, x0 = coords[0]
        if (y0, x0) not in visited:
            plastics.append(int(grid[y0, x0]))
            visited.add((y0, x0))

        for i in range(1, len(coords)):
            y1, x1 = coords[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                return False, f"Day {d} step {i} out of bounds: {(y1, x1)}", [], [], []
            dy, dx = y1 - y0, x1 - x0
            dir_idx = DIR_VECTORS.index((dy, dx))
            step_len = STEP_LENGTH[dir_idx]
            if dist + step_len > max_distance:
                return False, f"Day {d} exceeds {max_distance} km (stopped at step {i}).", [], [], []
            dist += step_len
            steps.append(step_len)
            if (y1, x1) not in visited:
                plastics.append(int(grid[y1, x1]))
                visited.add((y1, x1))
            y0, x0 = y1, x1

        plastic_by_day.append(plastics)
        distance_by_day.append(dist)
        distance_by_day_steps.append(steps)
        prev_end = coords[-1]
        prev_dir = new_dir

    return True, "Route validated successfully", plastic_by_day, distance_by_day, distance_by_day_steps


# ---------------------------------------------------------------------
# Draw Excel-style heatmap
# ---------------------------------------------------------------------
def draw_last_frame(grid, rotation_days, start_cell, start_dir,
                    plastic_by_day, distance_by_day_steps):
    n_rows, n_cols = grid.shape

    # Column letters (A, B, ..., AD)
    col_labels = []
    for i in range(n_cols):
        div, mod = divmod(i, 26)
        label = chr(65 + mod)
        if div > 0:
            label = chr(64 + div) + label
        col_labels.append(label)
    # Row numbers
    row_labels = [str(i + 1) for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        grid,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={"size": 16, "weight": "bold", "color": "black"}  # larger, bold numbers
    )

    # Excel layout
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cmap = plt.get_cmap("tab10")
    day_color_map = {i: cmap(i % 10) for i in range(len(rotation_days))}

    move_counter = 0
    y, x = start_cell
    dir_idx = DIR_TO_IDX[start_dir]

    for day_idx, rotations in enumerate(rotation_days):
        color = day_color_map[day_idx]
        coords, dir_idx = rotations_to_coords((y, x), IDX_TO_DIR[dir_idx], rotations)

        for j in range(1, len(coords)):
            (y0, x0), (y1, x1) = coords[j - 1], coords[j]
            x0c, y0c = x0 + 0.5, y0 + 0.5
            x1c, y1c = x1 + 0.5, y1 + 0.5
            xm, ym = (x0c + x1c) / 2, (y0c + y1c) / 2

            ax.annotate("", xy=(xm, ym), xytext=(x0c, y0c),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=2)

            move_counter += 1
            ax.text(x0c, y0c + 0.25, str(move_counter),
                    color="black", fontsize=8, ha="center", va="center",
                    weight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color,
                              linewidth=0.8, alpha=0.9))

            rect = patches.FancyBboxPatch(
                (x0, y0), 1, 1,
                boxstyle="round,pad=0.002,rounding_size=0.15",
                linewidth=3, edgecolor=color, facecolor="none",
                alpha=0.8, zorder=3 + day_idx
            )
            ax.add_patch(rect)

        y, x = coords[-1]

    # Start and End highlights
    y_start, x_start = start_cell
    start_rect = patches.FancyBboxPatch(
        (x_start, y_start), 1, 1,
        boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=3, edgecolor="green",
        facecolor="none", alpha=0.8, zorder=10
    )
    ax.add_patch(start_rect)

    last_y, last_x = y, x
    last_color = day_color_map[len(rotation_days) - 1]
    end_rect = patches.FancyBboxPatch(
        (last_x, last_y), 1, 1,
        boxstyle="round,pad=0.002,rounding_size=0.15",
        linewidth=9, edgecolor=last_color,
        facecolor="none", alpha=0.8, zorder=12
    )
    ax.add_patch(end_rect)

    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in distance_by_day_steps)
    ax.set_title(f"plastic = {plastic_total}    |    distance = {distance_total}",
                 fontsize=13, family="monospace", pad=15)

    legend_handles = [
        patches.Patch(color=day_color_map[i], label=f"Day {i + 1}")
        for i in range(len(rotation_days))
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    pdf_bytes = buf.read()
    buf.close()
    return fig, pdf_bytes


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("Validate Rotation-Code Path for The Ocean Cleanup Challenge")

example = "0, 0, 1, 0, -1\n0, 1, 1, 0\n0, 0, 0"
path_str = st.text_area("Enter rotation codes per day (use -1, 0, 1):", example)
start_y = st.number_input("Start row (y):", min_value=0, max_value=GRID.shape[0] - 1, value=0)
start_x = st.number_input("Start column (x):", min_value=0, max_value=GRID.shape[1] - 1, value=0)
start_dir = st.selectbox("Start direction:", list(DIR_TO_IDX.keys()), index=2)
max_days = st.number_input("Max days:", min_value=1, max_value=10, value=5)
max_distance = st.number_input("Max distance per day (km):", min_value=5, max_value=50, value=50)

if st.button("Validate and Draw"):
    try:
        rotation_days = parse_rotation_paths(path_str)
        n_days = len(rotation_days)
        st.info(f"🧭 Parsed {n_days} day{'s' if n_days > 1 else ''}.")
    except Exception as e:
        st.error(f"❌ Parsing error: {e}")
        st.stop()

    ok, msg, plastic_by_day, distance_by_day, distance_by_day_steps = validate_rotation_paths(
        GRID, rotation_days, (start_y, start_x), start_dir, max_days, max_distance
    )

    if ok:
        total_plastic = sum(sum(p) for p in plastic_by_day)
        total_distance = sum(sum(d) for d in distance_by_day_steps)
        avg_distance = np.mean(distance_by_day) if distance_by_day else 0

        st.success("✅ Route valid and successfully parsed!")
        st.markdown(
            f"### 📊 Key Performance Indicators\n"
            f"- **Days parsed:** {len(rotation_days)}  \n"
            f"- **Total plastic collected:** 🟢 {total_plastic} units  \n"
            f"- **Total distance traveled:** 🔵 {total_distance} km  \n"
            f"- **Average distance per day:** {avg_distance:.1f} km"
        )

        for d, (plastics, dist_steps) in enumerate(zip(plastic_by_day, distance_by_day_steps), start=1):
            st.markdown(
                f"**Day {d}:** Plastic = {sum(plastics)} ({'+'.join(map(str, plastics))})  |  "
                f"Distance = {sum(dist_steps)} ({'+'.join(map(str, dist_steps))})"
            )

        fig, pdf_bytes = draw_last_frame(
            GRID,
            rotation_days,
            (start_y, start_x),
            start_dir,
            plastic_by_day,
            distance_by_day_steps
        )
        st.pyplot(fig, clear_figure=True)
        st.download_button(
            "Download last frame as PDF",
            pdf_bytes,
            "last_frame.pdf",
            "application/pdf"
        )
    else:
        st.error("❌ Invalid route!")
        st.markdown(f"**Reason:** {msg}")
        st.info("Check your rotations: ensure the path stays inside the grid and daily distance ≤ max distance.")
