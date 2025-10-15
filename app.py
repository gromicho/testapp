# app.py
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
# Simple validation: check the path fits inside grid bounds
# ---------------------------------------------------------------------
def validate_path(day_paths, grid):
    """Minimal validation: coordinates in bounds."""
    rows, cols = grid.shape
    for day in day_paths:
        for (y, x) in day:
            if not (0 <= y < rows and 0 <= x < cols):
                return False
    return True


# ---------------------------------------------------------------------
# Plot the heatmap grid with Seaborn
# ---------------------------------------------------------------------
def plot_grid(grid):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(grid, annot=True, fmt='d', cmap='YlGnBu', square=True, ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title('Validate and Plot Grid')

example = '[[[0,0],[0,1],[0,2]], [[4,4],[4,5]]]'
path_str = st.text_area('Enter day_paths (list of lists of [y,x] coords):', example)

if st.button('Validate and Plot'):
    try:
        day_paths = eval(path_str)
        if validate_path(day_paths, GRID):
            st.success('✅ Path is valid.')
            plot_grid(GRID)
        else:
            st.error('❌ Invalid path: some coordinates are out of bounds.')
    except Exception as e:
        st.error(f'⚠️ Error parsing input: {e}')
