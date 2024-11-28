"""Utilities for the exercises."""

import numpy as np


def generate_2d_grid(Nx):
    """Adapted from Section 4.4."""
    Nnodes = Nx + 1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x, y)
    nodes = np.zeros((Nnodes**2, 2))
    nodes[:, 0] = X.ravel()
    nodes[:, 1] = Y.ravel()
    boundaries = {"dirichlet": [], "neumann": []}  # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 0):
            boundaries["dirichlet"].append(nID)  # Dirichlet BC
        else:
            n_eq += 1
            if (
                (np.allclose(nodes[nID, 1], 0))
                or (np.allclose(nodes[nID, 0], 1))
                or (np.allclose(nodes[nID, 1], 1))
            ):
                boundaries["neumann"].append(nID)  # Neumann BC
    IEN = np.zeros((2 * Nx**2, 3), dtype=np.int64)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2 * i + 2 * j * Nx, :] = (
                i + j * Nnodes,
                i + 1 + j * Nnodes,
                i + (j + 1) * Nnodes,
            )
            IEN[2 * i + 1 + 2 * j * Nx, :] = (
                i + 1 + j * Nnodes,
                i + 1 + (j + 1) * Nnodes,
                i + (j + 1) * Nnodes,
            )
    return nodes, IEN, boundaries
