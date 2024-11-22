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
