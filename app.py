# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import ast
import re

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
# Richtingen en staplengtes
# ---------------------------------------------------------------------
DIR_VECTORS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]
STEP_LENGTH = [5, 7, 5, 7, 5, 7, 5, 7]
DIR_TO_IDX = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}
IDX_TO_DIR = {v: k for k, v in DIR_TO_IDX.items()}

# ---------------------------------------------------------------------
# Excel-hulpfuncties
# ---------------------------------------------------------------------
def excel_to_coord(cell_ref):
    cell_ref = cell_ref.strip().upper()
    letters = ''.join(ch for ch in cell_ref if ch.isalpha())
    digits = ''.join(ch for ch in cell_ref if ch.isdigit())
    x = sum((ord(ch) - 64) * (26 ** i) for i, ch in enumerate(letters[::-1])) - 1
    y = int(digits) - 1
    return (y, x)

def coord_to_excel(y, x):
    x += 1
    letters = ''
    while x > 0:
        x, remainder = divmod(x - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{y + 1}"

def expand_range(start_ref, end_ref):
    y1, x1 = excel_to_coord(start_ref)
    y2, x2 = excel_to_coord(end_ref)
    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        return [(y1, x) for x in range(x1, x2 + step, step)]
    elif x1 == x2:
        step = 1 if y2 >= y1 else -1
        return [(y, x1) for y in range(y1, y2 + step, step)]
    else:
        raise ValueError(f"Reeks {start_ref}:{end_ref} is niet rechtlijnig.")

# ---------------------------------------------------------------------
# Parser: detecteert Excel of rotatie-invoer
# ---------------------------------------------------------------------
def parse_input_auto(text):
    s = text.strip()
    if not s:
        raise ValueError("Lege invoer.")
    if re.search(r'[A-Za-z]', s):
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        all_days = []
        for ln in lines:
            parts = re.split(r'[,;\s]+', ln)
            coords = []
            for p in parts:
                if not p:
                    continue
                if ':' in p:
                    start, end = p.split(':')
                    coords.extend(expand_range(start.strip(), end.strip()))
                else:
                    coords.append(excel_to_coord(p))
            all_days.append(coords)
        return 'excel', all_days
    else:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        all_days = []
        for ln in lines:
            vals = [int(v) for v in re.split(r'[,;\s]+', ln) if v]
            if not all(v in (-1, 0, 1) for v in vals):
                raise ValueError(f"Ongeldige rotatiecode in regel: {ln}")
            all_days.append(vals)
        return 'rotation', all_days

# ---------------------------------------------------------------------
# Rotaties naar coördinaten
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
# Validatie
# ---------------------------------------------------------------------
def validate_paths(grid, day_paths, start_cell, start_dir, max_days, max_distance, mode):
    rows, cols = grid.shape
    visited = set()
    plastic_by_day, distance_by_day, distance_by_day_steps = [], [], []
    y, x = start_cell
    if not (0 <= y < rows and 0 <= x < cols):
        return False, f"Startcel {coord_to_excel(y, x)} buiten raster.", [], [], []
    dir_idx = DIR_TO_IDX[start_dir]
    visited.add(start_cell)
    prev_end, prev_dir = start_cell, dir_idx

    for d, day in enumerate(day_paths, start=1):
        if not day:
            return False, f"Dag {d} is leeg.", [], [], []
        if mode == 'rotation':
            coords, new_dir = rotations_to_coords(prev_end, IDX_TO_DIR[prev_dir], day)
        else:
            coords, new_dir = day, prev_dir

        dist, plastics, steps = 0, [], []
        y0, x0 = coords[0]
        if (y0, x0) not in visited:
            plastics.append(int(grid[y0, x0]))
            visited.add((y0, x0))
        prev_step_dir = prev_dir

        for i in range(1, len(coords)):
            y1, x1 = coords[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                return False, f"Dag {d} stap {i} buiten raster: {coord_to_excel(y1, x1)}.", [], [], []
            dy, dx = y1 - y0, x1 - x0
            if (dy, dx) not in DIR_VECTORS:
                return False, f"Dag {d} stap {i} ongeldig (geen toegestane richting).", [], [], []
            dir_idx = DIR_VECTORS.index((dy, dx))

            # ✅ 45° beperking
            turn = (dir_idx - prev_step_dir) % 8
            if turn not in (0, 1, 7):
                return False, f"Dag {d} stap {i} maakt een verboden bocht (>45°).", [], [], []

            step_len = STEP_LENGTH[dir_idx]
            if dist + step_len > max_distance:
                return False, f"Dag {d} overschrijdt {max_distance} km bij stap {i}.", [], [], []
            dist += step_len
            steps.append(step_len)

            if (y1, x1) not in visited:
                plastics.append(int(grid[y1, x1]))
                visited.add((y1, x1))

            y0, x0 = y1, x1
            prev_step_dir = dir_idx

        plastic_by_day.append(plastics)
        distance_by_day.append(dist)
        distance_by_day_steps.append(steps)
        prev_end, prev_dir = coords[-1], prev_step_dir

    return True, "Route is geldig.", plastic_by_day, distance_by_day, distance_by_day_steps

# ---------------------------------------------------------------------
# Visualisatie
# ---------------------------------------------------------------------
def draw_last_frame(grid, day_paths, start_cell, start_dir, plastic_by_day, distance_by_day_steps):
    n_rows, n_cols = grid.shape
    col_labels = []
    for i in range(n_cols):
        div, mod = divmod(i, 26)
        label = chr(65 + mod)
        if div > 0:
            label = chr(64 + div) + label
        col_labels.append(label)
    row_labels = [str(i + 1) for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(grid, ax=ax, cmap="YlGnBu", annot=True, fmt="d",
                cbar=False, square=True,
                xticklabels=col_labels, yticklabels=row_labels,
                annot_kws={"size": 16, "weight": "bold"})

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("")
    ax.set_ylabel("")

    cmap = plt.get_cmap("tab10")
    day_color_map = {i: cmap(i % 10) for i in range(len(day_paths))}
    move_counter = 0

    for day_idx, day in enumerate(day_paths):
        color = day_color_map[day_idx]
        for j in range(1, len(day)):
            (y0, x0), (y1, x1) = day[j - 1], day[j]
            x0c, y0c = x0 + 0.5, y0 + 0.5
            x1c, y1c = x1 + 0.5, y1 + 0.5
            xm, ym = (x0c + x1c) / 2, (y0c + y1c) / 2
            ax.annotate("", xy=(xm, ym), xytext=(x0c, y0c),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2),
                        zorder=3)
            move_counter += 1
            ax.text(x0c, y0c + 0.25, str(move_counter),
                    fontsize=8, ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor=color,
                              linewidth=0.8, alpha=0.9), zorder=6)
            ax.add_patch(patches.FancyBboxPatch(
                (x0, y0), 1, 1,
                boxstyle="round,pad=0.002,rounding_size=0.15",
                linewidth=3, edgecolor=color, facecolor="none",
                alpha=0.8, zorder=4 + day_idx))

    y_start, x_start = start_cell
    ax.add_patch(patches.FancyBboxPatch((x_start, y_start), 1, 1,
                                        boxstyle="round,pad=0.002,rounding_size=0.15",
                                        linewidth=3, edgecolor="green",
                                        facecolor="none", alpha=0.8, zorder=10))
    last_y, last_x = day_paths[-1][-1]
    last_color = day_color_map[len(day_paths) - 1]
    ax.add_patch(patches.FancyBboxPatch((last_x, last_y), 1, 1,
                                        boxstyle="round,pad=0.002,rounding_size=0.15",
                                        linewidth=9, edgecolor=last_color,
                                        facecolor="none", alpha=0.8, zorder=12))
    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in distance_by_day_steps)
    ax.set_title(f"plastic = {plastic_total}    |    distance = {distance_total}",
                 fontsize=13, family="monospace", pad=15)

    legend_handles = [patches.Patch(color=day_color_map[i], label=f"Dag {i + 1}")
                      for i in range(len(day_paths))]
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return fig, buf.read()

# ---------------------------------------------------------------------
# Streamlit UI (NL)
# ---------------------------------------------------------------------
st.title("Validatie en Visualisatie van Routes – The Ocean Cleanup Challenge")

st.markdown("""
Gebruik dit hulpmiddel om je route te controleren:
- 🔄 **Rotatiecodes:** regels met `-1`, `0`, `1`
- 📘 **Excel-cellen:** zoals `B3:E3`, `E3:E6`

Elke regel stelt één dag voor. Alleen bochten van maximaal ±45° zijn toegestaan.
""")

example = "B3:E3\nE3:F4\nF4:G5"
path_str = st.text_area("Voer de route in:", example)
start_y = st.number_input("Start-rij (y):", 0, GRID.shape[0] - 1, 0)
start_x = st.number_input("Start-kolom (x):", 0, GRID.shape[1] - 1, 0)
start_dir = st.selectbox("Start-richting:", list(DIR_TO_IDX.keys()), index=2)
max_days = st.number_input("Maximaal aantal dagen:", 1, 10, 5)
max_distance = st.number_input("Maximale afstand per dag (km):", 5, 50, 50)

if st.button("Valideer en visualiseer"):
    try:
        mode, parsed = parse_input_auto(path_str)
        st.info(f"🔍 Herkend als {'Excel-positie' if mode == 'excel' else 'rotatie'}-invoer.")
    except Exception as e:
        st.error(f"Fout bij het inlezen: {e}")
        st.stop()

    ok, msg, plastic_by_day, dist_by_day, dist_steps = validate_paths(
        GRID, parsed, (start_y, start_x), start_dir, max_days, max_distance, mode
    )

    if ok:
        st.success("✅ Route geldig")
        total_plastic = sum(sum(p) for p in plastic_by_day)
        total_distance = sum(sum(d) for d in dist_steps)
        st.markdown(f"""
        ### 📊 Prestatie-indicatoren
        - Dagen: {len(parsed)}
        - Totaal plastic: **{total_plastic}**
        - Totale afstand: **{total_distance} km**
        """)
        fig, pdf_bytes = draw_last_frame(GRID, parsed, (start_y, start_x), start_dir,
                                         plastic_by_day, dist_steps)
        st.pyplot(fig, clear_figure=True)
        st.download_button("📥 Download als PDF", pdf_bytes, "route.pdf", "application/pdf")
    else:
        st.error("❌ Ongeldige route")
        st.warning(msg)
