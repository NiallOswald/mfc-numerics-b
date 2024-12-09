"""Utilities for the finite element method."""

import numpy as np
from scipy.optimize import linprog

np.seterr(invalid="ignore", divide="ignore")


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
    val = (
        amplitude
        * np.e
        * np.exp(-1 / (1 - np.linalg.norm((x - x0) / radius, axis=1) ** order))
    )
    val[np.linalg.norm(x - x0, axis=1) > radius] = 0.0
    return np.nan_to_num(val, nan=0.0)


def find_element(x, nodes, node_map):
    """Find the element containing the point x."""
    for i, element in enumerate(node_map):
        if is_inside(x, nodes[element]):
            return i
    else:
        raise ValueError("Point not found in any element.")


def is_inside(x, points):
    """Check if the point x is inside the convex hull of the points."""
    A_eq = np.vstack([points.T, np.ones(len(points))])
    b_eq = np.array([*x, 1])
    cost = np.zeros(len(points))
    return linprog(cost, A_eq=A_eq, b_eq=b_eq).success
