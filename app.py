# app.py
# Python 3.12 — ASCII only, PEP8 compliant.

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import re

# ---------------------------------------------------------------------
# Types (no from typing import ... to respect your preference)
# ---------------------------------------------------------------------
Coord = tuple[int, int]
DayCoords = list[Coord]
DaysCoords = list[DayCoords]
Rotations = list[int]
DaysRotations = list[Rotations]

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
DIR_VECTORS: list[Coord] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]
STEP_LENGTH: list[int] = [5, 7, 5, 7, 5, 7, 5, 7]
DIR_TO_IDX: dict[str, int] = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
IDX_TO_DIR: dict[int, str] = {v: k for k, v in DIR_TO_IDX.items()}

# ---------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------
def excel_to_coord(cell_ref: str) -> Coord:
    """
    Convert an Excel cell like 'B3' to zero-based (row, col) tuple (y, x).
    """
    s = cell_ref.strip().upper()
    letters = ''.join(ch for ch in s if ch.isalpha())
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not letters or not digits:
        raise ValueError(f'Onjuiste cel: {cell_ref}')
    x = sum((ord(ch) - 64) * (26 ** i) for i, ch in enumerate(letters[::-1])) - 1
    y = int(digits) - 1
    return (y, x)


def coord_to_excel(y: int, x: int) -> str:
    """
    Convert zero-based (row, col) to Excel label like 'B3'.
    """
    col = x + 1
    letters: list[str] = []
    while col > 0:
        col, r = divmod(col - 1, 26)
        letters.append(chr(65 + r))
    return ''.join(reversed(letters)) + str(y + 1)


def expand_range(start_ref: str, end_ref: str) -> DayCoords:
    """
    Expand a straight Excel range to a list of (y, x) cells.
    Supports horizontal or vertical straight ranges only.
    """
    y1, x1 = excel_to_coord(start_ref)
    y2, x2 = excel_to_coord(end_ref)
    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        return [(y1, x) for x in range(x1, x2 + step, step)]
    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        return [(y, x1) for y in range(y1, y2 + step, step)]
    raise ValueError(f'Reeks {start_ref}:{end_ref} is niet rechtlijnig.')


def make_excel_labels(n_cols: int) -> list[str]:
    """
    Generate Excel-style column headers: A, B, ..., Z, AA, AB, ...
    """
    labels: list[str] = []
    for i in range(n_cols):
        col = i + 1
        letters: list[str] = []
        while col > 0:
            col, r = divmod(col - 1, 26)
            letters.append(chr(65 + r))
        labels.append(''.join(reversed(letters)))
    return labels

# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------
def parse_input_auto(text: str) -> tuple[str, list]:
    """
    Detect Excel- or rotation-style input, return ('excel'|'rotation', parsed).
    Excel: each line contains cells and ranges like A1, B2:D2.
    Rotation: each line contains ints chosen from -1, 0, 1.
    """
    s = text.strip()
    if not s:
        raise ValueError('Lege invoer.')
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

    # Basic heuristic: any letter means Excel mode
    if re.search(r'[A-Za-z]', s):
        all_days: DaysCoords = []
        for ln in lines:
            parts = re.split(r'[,;\s]+', ln)
            coords: DayCoords = []
            for p in parts:
                if not p:
                    continue
                if ':' in p:
                    start, end = p.split(':', 1)
                    coords.extend(expand_range(start.strip(), end.strip()))
                else:
                    coords.append(excel_to_coord(p))
            if not coords:
                raise ValueError('Lege regel in Excel-invoer.')
            all_days.append(coords)
        return 'excel', all_days

    # Otherwise rotation mode
    all_days_rot: DaysRotations = []
    for ln in lines:
        vals = [int(v) for v in re.split(r'[,;\s]+', ln) if v]
        if not vals:
            raise ValueError('Lege regel in rotatie-invoer.')
        if not all(v in (-1, 0, 1) for v in vals):
            raise ValueError(f'Ongeldige rotatiecode in regel: {ln}')
        all_days_rot.append(vals)
    return 'rotation', all_days_rot

# ---------------------------------------------------------------------
# Rotation to coordinates
# ---------------------------------------------------------------------
def rotations_to_coords(start_cell: Coord, start_dir: str, rotations: Rotations) -> tuple[DayCoords, int]:
    """
    Build coordinate sequence from start cell, heading, and turns.
    Returns (coords including start, final_dir_idx).
    """
    y, x = start_cell
    dir_idx = DIR_TO_IDX[start_dir]
    coords: DayCoords = [(y, x)]
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
    day_paths_raw: list,
    start_cell: Coord,
    start_dir: str,
    max_days: int,
    max_distance: int,
    mode: str
) -> tuple[
    bool,
    list[str],
    list[list[int]],
    list[int],
    list[list[int]],
    list[list[dict]]
]:
    """
    Validate daily paths. Returns rich diagnostics and KPIs.
    Tracks both per-day and global cumulative plastic.
    """
    rows, cols = grid.shape
    messages: list[str] = []
    visited: set[Coord] = set()
    plastic_by_day: list[list[int]] = []
    distance_by_day: list[int] = []
    distance_by_day_steps: list[list[int]] = []
    step_logs_all_days: list[list[dict]] = []

    y, x = start_cell
    if not (0 <= y < rows and 0 <= x < cols):
        return False, [f'Startcel {coord_to_excel(y, x)} ligt buiten het raster.'], [], [], [], []
    dir_idx_global = DIR_TO_IDX[start_dir]
    visited.add(start_cell)
    prev_end, prev_dir_idx = start_cell, dir_idx_global

    if len(day_paths_raw) > max_days:
        return False, [f'Aantal opgegeven dagen ({len(day_paths_raw)}) overschrijdt maximum ({max_days}).'], [], [], [], []

    global_plastic_cum = 0  # totale cumulatieve teller over alle dagen

    for d_idx, day_raw in enumerate(day_paths_raw, start=1):
        if not day_raw:
            return False, [f'Dag {d_idx} is leeg.'], [], [], [], []

        if mode == 'rotation':
            coords, new_dir_idx = rotations_to_coords(prev_end, IDX_TO_DIR[prev_dir_idx], day_raw)
        else:
            coords = day_raw
            new_dir_idx = prev_dir_idx

        day_dist = 0
        day_plastics: list[int] = []
        day_step_lengths: list[int] = []
        day_logs: list[dict] = []

        y0, x0 = coords[0]
        if not (0 <= y0 < rows and 0 <= x0 < cols):
            return False, [f'Dag {d_idx} start buiten raster: {coord_to_excel(y0, x0)}.'], [], [], [], []

        if (y0, x0) not in visited:
            gain = int(grid[y0, x0])
            day_plastics.append(gain)
            visited.add((y0, x0))
            global_plastic_cum += gain
        else:
            gain = 0

        # day_logs.append({
        #     'step_no': 0,
        #     'from': coord_to_excel(y0, x0),
        #     'to': coord_to_excel(y0, x0),
        #     'dir': IDX_TO_DIR[prev_dir_idx],
        #     'turn': 'start',
        #     'step_km': 0,
        #     'cum_km': 0,
        #     'plastic_gain': gain,
        #     'plastic_cum_day': sum(day_plastics),
        #     'plastic_cum_total': global_plastic_cum,
        #     'revisit': gain == 0
        # })

        prev_step_dir_idx = prev_dir_idx
        for i in range(1, len(coords)):
            y1, x1 = coords[i]
            if not (0 <= y1 < rows and 0 <= x1 < cols):
                fail_cell = coord_to_excel(y1, x1)
                from_cell = coord_to_excel(y0, x0)
                reason = f'Dag {d_idx} stap {i}: {from_cell} -> {fail_cell} valt buiten het raster.'
                return False, [reason], plastic_by_day, distance_by_day, distance_by_day_steps, step_logs_all_days + [day_logs]

            dy, dx = y1 - y0, x1 - x0
            if (dy, dx) not in DIR_VECTORS:
                reason = f'Dag {d_idx} stap {i}: geen toegestane richting van {coord_to_excel(y0, x0)} naar {coord_to_excel(y1, x1)}.'
                return False, [reason], plastic_by_day, distance_by_day, distance_by_day_steps, step_logs_all_days + [day_logs]

            dir_idx = DIR_VECTORS.index((dy, dx))
            turn_rel = (dir_idx - prev_step_dir_idx) % 8
            if turn_rel not in (0, 1, 7):
                reason = (
                    f'Dag {d_idx} stap {i}: verboden bocht (>45 graden) '
                    f'van {IDX_TO_DIR[prev_step_dir_idx]} naar {IDX_TO_DIR[dir_idx]} '
                    f'bij {coord_to_excel(y0, x0)} -> {coord_to_excel(y1, x1)}.'
                )
                return False, [reason], plastic_by_day, distance_by_day, distance_by_day_steps, step_logs_all_days + [day_logs]

            step_len = STEP_LENGTH[dir_idx]
            if day_dist + step_len > max_distance:
                reason = (
                    f'Dag {d_idx} stap {i}: daglimiet {max_distance} km overschreden '
                    f'({day_dist} + {step_len} km) bij verplaatsing '
                    f'{coord_to_excel(y0, x0)} -> {coord_to_excel(y1, x1)}.'
                )
                return False, [reason], plastic_by_day, distance_by_day, distance_by_day_steps, step_logs_all_days + [day_logs]

            day_dist += step_len
            day_step_lengths.append(step_len)

            revisit = (y1, x1) in visited
            gain = 0 if revisit else int(grid[y1, x1])
            if not revisit:
                day_plastics.append(gain)
                visited.add((y1, x1))
                global_plastic_cum += gain

            day_logs.append({
                'step_no': i,
                'from': coord_to_excel(y0, x0),
                'to': coord_to_excel(y1, x1),
                'dir': IDX_TO_DIR[dir_idx],
                'turn': 'right' if turn_rel == 1 else ('left' if turn_rel == 7 else 'straight'),
                'step_km': step_len,
                'cum_km': day_dist,
                'plastic_gain': gain,
                'plastic_cum_day': sum(day_plastics),
                'plastic_cum_total': global_plastic_cum,
                'revisit': revisit
            })

            y0, x0 = y1, x1
            prev_step_dir_idx = dir_idx

        plastic_by_day.append(day_plastics)
        distance_by_day.append(day_dist)
        distance_by_day_steps.append(day_step_lengths)
        step_logs_all_days.append(day_logs)
        prev_end, prev_dir_idx = coords[-1], prev_step_dir_idx

    start_excel = coord_to_excel(*start_cell)
    messages.append(f'Route is geldig vanaf {start_excel}.')
    return True, messages, plastic_by_day, distance_by_day, distance_by_day_steps, step_logs_all_days


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def draw_last_frame(
    grid: np.ndarray,
    day_paths: DaysCoords,
    start_cell: Coord,
    start_dir: str,
    plastic_by_day: list[list[int]],
    distance_by_day_steps: list[list[int]]
) -> tuple[plt.Figure, bytes]:
    """
    Draw the final grid with the full route overlay and return (matplotlib_fig, pdf_bytes).
    """
    n_rows, n_cols = grid.shape
    col_labels = make_excel_labels(n_cols)
    row_labels = [str(i + 1) for i in range(n_rows)]

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        grid, ax=ax, cmap='YlGnBu', annot=True, fmt='d',
        cbar=False, square=True,
        xticklabels=col_labels, yticklabels=row_labels,
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
            xm, ym = (x0c + x1c) / 2.0, (y0c + y1c) / 2.0
            ax.annotate(
                '', xy=(xm, ym), xytext=(x0c, y0c),
                arrowprops=dict(arrowstyle='->', color=color, lw=2),
                zorder=3
            )
            move_counter += 1
            ax.text(
                x0c, y0c + 0.25, str(move_counter),
                fontsize=8, ha='center', va='center', weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='white', edgecolor=color,
                    linewidth=0.8, alpha=0.9
                ),
                zorder=6
            )
            ax.add_patch(patches.FancyBboxPatch(
                (x0, y0), 1, 1,
                boxstyle='round,pad=0.002,rounding_size=0.15',
                linewidth=3, edgecolor=color, facecolor='none',
                alpha=0.8, zorder=4 + day_idx
            ))

    # Start in green
    y_start, x_start = start_cell
    ax.add_patch(patches.FancyBboxPatch(
        (x_start, y_start), 1, 1,
        boxstyle='round,pad=0.002,rounding_size=0.15',
        linewidth=3, edgecolor='green',
        facecolor='none', alpha=0.8, zorder=10
    ))

    # Last cell highlighted
    last_y, last_x = day_paths[-1][-1]
    last_color = day_color_map[len(day_paths) - 1]
    ax.add_patch(patches.FancyBboxPatch(
        (last_x, last_y), 1, 1,
        boxstyle='round,pad=0.002,rounding_size=0.15',
        linewidth=9, edgecolor=last_color,
        facecolor='none', alpha=0.8, zorder=12
    ))

    plastic_total = sum(sum(p) for p in plastic_by_day)
    distance_total = sum(sum(d) for d in distance_by_day_steps)
    start_excel = coord_to_excel(*start_cell)
    ax.set_title(
        f'Start = {start_excel} | Plastic = {plastic_total} | Distance = {distance_total}',
        fontsize=13, family='monospace', pad=15
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
# Streamlit UI
# ---------------------------------------------------------------------
st.title('Validatie en Visualisatie van Routes — The Ocean Cleanup Challenge')

st.markdown(
    'Gebruik dit hulpmiddel om je route te controleren:\n'
    '- Rotatiecodes: regels met -1, 0, 1\n'
    '- Excel-cellen: zoals B3:E3, E3:E6\n\n'
    'Elke regel stelt één dag voor. Alleen bochten van maximaal 45 graden zijn toegestaan.'
)

example = 'B3:E3\nE3:F4\nF4:G5'
path_str = st.text_area('Voer de route in:', example)

col_labels = make_excel_labels(GRID.shape[1])
row_labels = [str(i + 1) for i in range(GRID.shape[0])]

start_col_letter = st.selectbox('Start-kolom:', col_labels, index=6)
start_row_label = st.selectbox('Start-rij:', row_labels, index=7)

start_x = col_labels.index(start_col_letter)
start_y = int(start_row_label) - 1
start_cell_excel = f'{start_col_letter}{start_row_label}'

start_dir = st.selectbox('Start-richting:', list(DIR_TO_IDX.keys()), index=2)
max_days = st.number_input('Maximaal aantal dagen:', min_value=1, max_value=10, value=5, step=1)
max_distance = st.number_input('Maximale afstand per dag (km):', min_value=5, max_value=50, value=50, step=1)

if st.button('Valideer en visualiseer'):
    try:
        mode, parsed = parse_input_auto(path_str)
        st.info(f'Herkend als {"Excel-positie" if mode == "excel" else "rotatie"}-invoer.')
    except Exception as e:
        st.error(f'Fout bij het inlezen: {e}')
        st.stop()

    ok, msgs, plastic_by_day, dist_by_day, dist_steps, step_logs = validate_paths(
        GRID, parsed, (start_y, start_x), start_dir, int(max_days), int(max_distance), mode
    )

    if not ok:
        st.error('Ongeldige route')
        for m in msgs:
            st.warning(m)
        # If we have partial logs for the failing day, show them to explain where it failed
        if step_logs:
            with st.expander('Details tot aan de fout'):
                for d_idx, logs in enumerate(step_logs, start=1):
                    st.markdown(f'**Dag {d_idx}**')
                    if not logs:
                        st.write('Geen stappen geregistreerd.')
                        continue
                    st.dataframe({
                        'step': [log['step_no'] for log in logs],
                        'from': [log['from'] for log in logs],
                        'to': [log['to'] for log in logs],
                        'dir': [log['dir'] for log in logs],
                        'turn': [log['turn'] for log in logs],
                        'step_km': [log['step_km'] for log in logs],
                        'cum_km': [log['cum_km'] for log in logs],
                        'plastic_gain': [log['plastic_gain'] for log in logs],
                        'plastic_cum': [log['plastic_cum'] for log in logs],
                        'revisit': [log['revisit'] for log in logs],
                    })
        st.stop()

    # Success path
    st.success('\n'.join(msgs))

    total_plastic = sum(sum(p) for p in plastic_by_day)
    total_distance = sum(sum(d) for d in dist_steps)
    st.markdown(
        f'### KPI overzicht\n'
        f'- Startcel: **{start_cell_excel}**\n'
        f'- Dagen: {len(parsed)}\n'
        f'- Totaal plastic: **{total_plastic}**\n'
        f'- Totale afstand: **{total_distance} km**'
    )

    # If rotation mode, convert to coordinates for drawing
    if mode == 'rotation':
        coords_paths: DaysCoords = []
        end_cell: Coord = (int(start_y), int(start_x))
        end_dir = start_dir
        for rday in parsed:
            coords, new_dir = rotations_to_coords(end_cell, end_dir, rday)
            coords_paths.append(coords)
            end_cell, end_dir = coords[-1], IDX_TO_DIR[new_dir]
    else:
        coords_paths = parsed  # already coordinates

    # Day-level KPIs
    with st.expander('Dagelijkse KPI\'s'):
        for d_idx, (plastics, km, steps, logs) in enumerate(
            zip(plastic_by_day, dist_by_day, dist_steps, step_logs), start=1
        ):
            st.markdown(f'**Dag {d_idx}** — plastic: {sum(plastics)}, afstand: {km} km, stappen: {len(steps)}')
            # Step-level table
            st.dataframe({
                'step': [log['step_no'] for log in logs],
                'from': [log['from'] for log in logs],
                'to': [log['to'] for log in logs],
                'dir': [log['dir'] for log in logs],
                'turn': [log['turn'] for log in logs],
                'dist': [log['step_km'] for log in logs],
                'km': [log['cum_km'] for log in logs],
                'gain': [log['plastic_gain'] for log in logs],
                'day': [log['plastic_cum_day'] for log in logs],
                'plastic': [log['plastic_cum_total'] for log in logs],
                'revisit': [log['revisit'] for log in logs],
            })

    # Visualization
    fig, pdf_bytes = draw_last_frame(
        GRID, coords_paths, (int(start_y), int(start_x)),
        start_dir, plastic_by_day, dist_steps
    )
    st.pyplot(fig, clear_figure=True)
    pdf_filename = f'route_{start_cell_excel}.pdf'
    st.download_button('Download als PDF', pdf_bytes, pdf_filename, 'application/pdf')
