"""Utilities for the finite element method."""

import numpy as np


def load_mesh(name: str, scale: str):
    """Load the mesh from a file."""
    nodes = np.loadtxt(f"mesh/{name}_grids/{name}_nodes_{scale}.txt")
    node_map = np.loadtxt(f"mesh/{name}_grids/{name}_IEN_{scale}.txt", dtype=np.int64)
    boundary_nodes = np.loadtxt(
        f"mesh/{name}_grids/{name}_bdry_{scale}.txt", dtype=np.int64
    )

    return nodes, node_map, boundary_nodes


def gaussian_source(x, x0, amplitude=1.0, radius=1.0, order=2.0):
    """A gaussian source term."""
    val = amplitude * np.exp(
        -1 / (radius**order - np.linalg.norm(x - x0, axis=1) ** order)
    )
    val[np.linalg.norm(x - x0, axis=1) > radius] = 0.0
    return np.nan_to_num(val, nan=0.0)
