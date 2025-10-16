# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import re
from typing import List, Tuple, Literal, Dict  # no "from typing import ..." import was requested to be avoided, but type hints help editors

# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
Coord = Tuple[int, int]
Mode = Literal['excel', 'rotation']

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
# Directions and step lengths
# ---------------------------------------------------------------------
DIR_VECTORS: List[Coord] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]
STEP_LENGTH: List[int] = [5, 7, 5, 7, 5, 7, 5, 7]
DIR_TO_IDX: Dict[str, int] = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
IDX_TO_DIR: Dict[int, str] = {v: k for k, v in DIR_TO_IDX.items()}

# ---------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------
def excel_to_coord(cell_ref: str) -> Coord:
    """Convert an Excel cell label like 'B3' to (row, col) zero-based."""
    s = cell_ref.strip().upper()
    if not s or not re.search(r'^[A-Z]+[0-9]+$', s):
        raise ValueError(f'Invalid cell reference: {cell_ref}')
    letters = ''.join(ch for ch in s if ch.isalpha())
    digits = ''.join(ch for ch in s if ch.isdigit())
    x = sum((ord(ch) - 64) * (26 ** i) for i, ch in enumerate(letters[::-1])) - 1
    y = int(digits) - 1
    return (y, x)


def coord_to_excel(y: int, x: int) -> str:
    """Convert zero-based (row, col) to an Excel label like 'B3'."""
    col = x + 1
    letters = []
    while col > 0:
        col, r = divmod(col - 1, 26)
        letters.append(chr(65 + r))
    return ''.join(reversed(letters)) + str(y + 1)


def expand_range(start_ref: str, end_ref: str) -> List[Coord]:
    """Expand a straight Excel range into a list of contiguous cells."""
    y1, x1 = excel_to_coord(start_ref)
    y2, x2 = excel_to_coord(end_ref)
    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        return [(y1, x) for x in range(x1, x2 + step, step)]
    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        return [(y, x1) for y in range(y1, y2 + step, step)]
    raise ValueError(f'Range {start_ref}:{end_ref} is not straight.')


def check_coords_in_grid(coords: List[Coord], grid: np.ndarray) -> None:
    """Raise if any coordinate is out of the grid bounds."""
    rows, cols = grid.shape
    for y, x in coords:
        if not (0 <= y < rows and 0 <= x < cols):
            raise ValueError(f'Cell {coord_to_excel(y, x)} out of grid bounds.')

# ---------------------------------------------------------------------
# Parser: detect Excel vs rotation input
# ---------------------------------------------------------------------
def parse_input_auto(text: str, grid: np.ndarray) -> Tuple[Mode, List]:
    """
    Detect input type. If any letter appears, treat as Excel cells/ranges.
    Otherwise expect lines of -1/0/1 for rotations.
    """
    s = text.strip()
    if not s:
        raise ValueError('Empty input.')
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if re.search(r'[A-Za-z]', s):
        all_days: List[List[Coord]] = []
        for ln in lines:
            parts = re.split(r'[,;\\s]+', ln)
            coords: List[Coord] = []
            for p in parts:
                if not p:
                    continue
                if ':' in p:
                    start, end = p.split(':', 1)
                    seg = expand_range(start.strip(), end.strip())
                    check_coords_in_grid(seg, grid)
                    coords.extend(seg)
                else:
                    c = excel_to_coord(p)
                    check_coords_in_grid([c], grid)
                    coords.append(c)
            all_days.append(coords)
        return 'excel', all_days
    else:
        all_days = []
        for ln in lines:
            vals = [int(v) for v in re.split(r'[,;\\s]+', ln) if v]
            if not all(v in (-1, 0, 1) for v in vals):
                raise ValueError(f'Invalid rotation codes in line: {ln}')
            all_days.append(vals)
        return 'rotation', all_days

# ---------------------------------------------------------------------
# Rotations to coordinates
# ---------------------------------------------------------------------
def rotations_to_coords(start_cell: Coord, start_dir: str, rotations: List[int]) -> Tuple[List[Coord], int]:
    """Build a polyline of cells from a start cell, start dir, and -1/0/1 turns."""
    y, x = start_cell
    dir_idx = DIR_TO_IDX[start_dir]
    coords: List[Coord] = [(y, x)]
    for r in rotations:
        dir_idx = (dir_idx + r) % 8
        dy, dx = DIR_VECTORS[dir_idx]
        y, x = y + dy, x + dx
        coords.append((y, x))
    return coords, dir_idx

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_paths(
    grid: np.ndarray,
    day_paths,
    start_cell: Coord,
    start_dir: str,
    max_days: int,
    max_distance: int,
    mode: Mode
) -> Tuple[bool, str, List[List[int]], List[int], List[List[int]]]:
    """
    Validate a multi-day path.
    Returns (ok, message, plastics_by_day, distance_by_day, step_lengths_by_day).
    """
    rows, cols = grid.shape
    if len(day_paths) > max_days:
        return False, f'Number of days exceeds maximum {max_days}.', [], [], []

    visited = set()
    plastic_by_day: List[List[int]] = []
    distance_by_day: List[int] = []
    step_lengths_by_day: List[List[int]] = []

    y, x = start_cell
    if not (0 <= y < rows and 0 <= x < cols):
        return False, f'Start cell {coord_to_excel(y, x)} out of grid.', [], [], []
    dir_idx = DIR_TO_IDX[start_dir]
    visited.add(start_cell)
    prev_end, prev_dir = start_cell, dir_idx

    for d, day in enumerate(day_paths, start=1):
        if not day:
            return False, f'Day {d} is empty.', [], [], []

        if mode == 'rotation':
            coords, new_dir = rotations_to_coords(prev_end, IDX_TO_DIR[prev_dir], day)
        else:
            coords, new_dir = day, prev_dir

        # Start-of-day
        dist = 0
        plastics: List[int] = []
        steps: List[int] = []

        y0, x0 = coords[0]
        if not (0 <= y0 < rows and 0 <= x0 < cols):
            return False, f'Day {d} step 0 out of grid: {coord_to_excel(y0, x0)}.', [], [], []

        if (y0, x0) not in visited:
            plastics.append(int(grid[y0, x0]))
            visited.add((y0, x0))

        prev_step_dir = prev_dir

        for i in range(1, len(coords)):
            y1, x1 = coords[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                return False, f'Day {d} step {i} out of grid: {coord_to_excel(y1, x1)}.', [], [], []

            dy, dx = y1 - y0, x1 - x0
            if (dy, dx) not in DIR_VECTORS:
                return False, f'Day {d} step {i} is not an allowed move.', [], [], []
            step_dir = DIR_VECTORS.index((dy, dx))

            # 45-degree turn check
            turn = (step_dir - prev_step_dir) % 8
            if turn not in (0, 1, 7):
                return False, f'Day {d} step {i} turns more than 45 degrees.', [], [], []

            step_len = STEP_LENGTH[step_dir]
            if dist + step_len > max_distance:
                return False, f'Day {d} exceeds {max_distance} km at step {i}.', [], [], []
            dist += step_len
            steps.append(step_len)

            if (y1, x1) not in visited:
                plastics.append(int(grid[y1, x1]))
                visited.add((y1, x1))

            y0, x0 = y1, x1
            prev_step_dir = step_dir

        plastic_by_day.append(plastics)
        distance_by_day.append(dist)
        step_lengths_by_day.append(steps)
        prev_end, prev_dir = coords[-1], prev_step_dir

    return True, 'Route is valid.', plastic_by_day, distance_by_day, step_lengths_by_day

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def build_excel_col_labels(n_cols: int) -> List[str]:
    """A, B, ..., Z, AA, AB, ..."""
    out = []
    for i in range(n_cols):
        col = i + 1
        letters = []
        while col > 0:
            col, r = divmod(col - 1, 26)
            letters.append(chr(65 + r))
        out.append(''.join(reversed(letters)))
    return out


def draw_last_frame(
    grid: np.ndarray,
    day_paths,
    start_cell: Coord,
    start_dir: str,
    plastic_by_day: List[List[int]],
    step_lengths_by_day: List[List[int]]
) -> Tuple[plt.Figure, bytes]:
    """Render heatmap and overlay paths, return (matplotlib figure, pdf bytes)."""
    n_rows, n_cols = grid.shape
    col_labels = build_excel_col_labels(n_cols)
    row_labels = [str(i + 1) for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        grid,
        ax=ax,
        cmap='YlGnBu',
        annot=True,
        fmt='d',
        cbar=False,
        square=True,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot_kws={'size': 16, 'weight': 'bold'}
    )

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('')
    ax.set_ylabel('')

    cmap = plt.get_cmap('tab10')
    day_color_map = {i: cmap(i % 10) for i in range(len(day_paths))}
    move_counter = 0

    for day_idx, day in enumerate(day_paths):
        color = day_color_map[day_idx]
        for j in range(1, len(day)):
            (y0, x0), (y1, x1) = day[j - 1], day[j]
            x0c, y0c = x0 + 0.5, y0 + 0.5
            x1c, y1c = x1 + 0.5, y1 + 0.5
            xm, ym = (x0c + x1c) / 2, (y0c + y1c) / 2
            ax.annotate(
                '',
                xy=(xm, ym),
                xytext=(x0c, y0c),
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                zorder=3
            )
            move_counter += 1
            ax.text(
                x0c,
                y0c + 0.25,
                str(move_counter),
                fontsize=8,
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='white',
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.9
                ),
                zorder=6
            )
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x0, y0),
                    1,
                    1,
                    boxstyle='round,pad=0.002,rounding_size=0.15',
                    linewidth=3,
                    edgecolor=color,
                    facecolor='none',
                    alpha=0.8,
                    zorder=4 + day_idx
                )
            )

    # Start cell in green
    y_start, x_start = start_cell
    ax.add_patch(
        patches.FancyBboxPatch(
            (x_start, y_start),
            1,
            1,
            boxstyle='round,pad=0.002,rounding_size=0.15',
            linewidth=3,
            edgecolor='green',
            facecolor='none',
            alpha=0.8,
            zorder=10
        )
    )

    # Last cell thick outline
    last_y, last_x = day_paths[-1][-1]
    last_color = day_color_map[len(day_paths) - 1]
    ax.add_patch(
        patches.FancyBboxPatch(
            (last_x, last_y),
            1,
            1,
            boxstyle='round,pad=0.002,rounding_size=0.15',
            linewidth=9,
            edgecolor=last_color,
            facecolor='none',
            alpha=0.8,
            zorder=12
        )
    )

    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in step_lengths_by_day)
    ax.set_title(
        f'plastic = {plastic_total} | distance = {distance_total}',
        fontsize=13,
        family='monospace',
        pad=15
    )

    legend_handles = [patches.Patch(color=day_color_map[i], label=f'Dag {i + 1}')
                      for i in range(len(day_paths))]
    ax.legend(handles=legend_handles, loc='center left',
              bbox_to_anchor=(1, 0.5), fontsize=10, frameon=False)

    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return fig, buf.read()

# ---------------------------------------------------------------------
# Streamlit UI (NL)
# ---------------------------------------------------------------------
st.title('Validatie en Visualisatie van Routes — The Ocean Cleanup Challenge')

st.markdown(
    'Gebruik dit hulpmiddel om je route te controleren:\n'
    '- Rotatiecodes: regels met -1, 0, 1\n'
    '- Excel-cellen: zoals B3:E3, E3:E6\n\n'
    'Elke regel stelt een dag voor. Bochten van maximaal +-45 graden zijn toegestaan.'
)

example = 'B3:E3\nE3:F4\nF4:G5'
path_str = st.text_area('Voer de route in:', example)

start_y = st.number_input(
    'Start-rij (y):',
    min_value=0,
    max_value=GRID.shape[0] - 1,
    value=0,
    step=1
)
start_x = st.number_input(
    'Start-kolom (x):',
    min_value=0,
    max_value=GRID.shape[1] - 1,
    value=0,
    step=1
)
start_dir = st.selectbox('Start-richting:', list(DIR_TO_IDX.keys()), index=2)

max_days = st.number_input('Maximaal aantal dagen:', min_value=1, max_value=10, value=5, step=1)
max_distance = st.number_input('Maximale afstand per dag (km):', min_value=5, max_value=50, value=50, step=1)

if st.button('Valideer en visualiseer'):
    try:
        mode, parsed = parse_input_auto(path_str, GRID)
        st.info(f'Herkend als {"Excel-positie" if mode == "excel" else "rotatie"}-invoer.')
    except Exception as e:
        st.error(f'Fout bij het inlezen: {e}')
        st.stop()

    ok, msg, plastic_by_day, dist_by_day, dist_steps = validate_paths(
        GRID, parsed, (start_y, start_x), start_dir, int(max_days), int(max_distance), mode
    )

    if ok:
        st.success('Route geldig')
        total_plastic = sum(sum(p) for p in plastic_by_day)
        total_distance = sum(sum(d) for d in dist_steps)
        st.markdown(
            f'### Prestatie-indicatoren\n'
            f'- Dagen: {len(parsed)}\n'
            f'- Totaal plastic: **{total_plastic}**\n'
            f'- Totale afstand: **{total_distance} km**'
        )
        fig, pdf_bytes = draw_last_frame(GRID, parsed, (int(start_y), int(start_x)), start_dir, plastic_by_day, dist_steps)
        st.pyplot(fig, clear_figure=True)
        st.download_button('Download als PDF', pdf_bytes, 'route.pdf', 'application/pdf')
    else:
        st.error('Ongeldige route')
        st.warning(msg)
