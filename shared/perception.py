"""Grid perception utilities: connected components, color histograms, deltas."""

import numpy as np
from collections import deque


def one_hot_grid(grid: np.ndarray, num_colors: int = 16, pad_to: int = 64) -> np.ndarray:
    """One-hot encode grid to (num_colors, pad_to, pad_to) float32 array."""
    h, w = grid.shape
    out = np.zeros((num_colors, pad_to, pad_to), dtype=np.float32)
    for c in range(num_colors):
        out[c, :h, :w] = (grid == c).astype(np.float32)
    return out


def find_background_color(grid: np.ndarray) -> int:
    """Most common color in the grid."""
    counts = np.bincount(grid.flatten(), minlength=16)
    return int(np.argmax(counts))


def connected_components(grid: np.ndarray, background: int = None):
    """Find connected components (flood fill). Returns list of object dicts."""
    h, w = grid.shape
    if background is None:
        background = find_background_color(grid)

    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or grid[y, x] == background:
                continue
            color = int(grid[y, x])
            component = []
            queue = deque([(x, y)])
            while queue:
                cx, cy = queue.popleft()
                if 0 <= cx < w and 0 <= cy < h and not visited[cy, cx] and grid[cy, cx] == color:
                    visited[cy, cx] = True
                    component.append((cx, cy))
                    queue.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
            if len(component) >= 1:
                xs, ys = zip(*component)
                objects.append({
                    "color": color,
                    "size": len(component),
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "center": [round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1)],
                    "pixels": component,
                })
    return objects


def compute_delta(grid: np.ndarray, prev_grid: np.ndarray) -> dict:
    """Compute difference between two grid states."""
    if prev_grid is None:
        return None
    changed = np.argwhere(grid != prev_grid)
    return {
        "num_changed": int(len(changed)),
        "changed_cells": [(int(c[1]), int(c[0])) for c in changed[:30]],
    }


def serialize_grid(grid: np.ndarray, prev_grid: np.ndarray = None) -> dict:
    """Convert raw grid to structured observation dict."""
    h, w = grid.shape
    unique, counts = np.unique(grid, return_counts=True)
    background = find_background_color(grid)
    objects = connected_components(grid, background)

    obs = {
        "grid_size": f"{w}x{h}",
        "background_color": background,
        "color_counts": {int(c): int(n) for c, n in zip(unique, counts)},
        "num_objects": len(objects),
        "objects": [
            {k: v for k, v in obj.items() if k != "pixels"}
            for obj in objects
        ],
        "delta": compute_delta(grid, prev_grid),
    }
    return obs


def grid_to_ascii(grid: np.ndarray) -> str:
    """Render grid as compact ASCII string for LLM consumption."""
    hex_chars = "0123456789abcdef"
    lines = []
    h, w = grid.shape
    for y in range(h):
        line = "".join(hex_chars[int(grid[y, x])] for x in range(w))
        lines.append(line)
    return "\n".join(lines)
