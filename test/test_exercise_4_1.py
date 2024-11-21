"""Tests for exercise 4.1."""

from exercises.exercise_4_1 import generate_2d_grid, poisson_2d
import numpy as np


def test_simple():
    """Test the FEM for a simple source term."""
    N = 25
    S = lambda x: 1.0
    exact_sol = lambda x: x[..., 0] * (1 - x[..., 0] / 2)

    mesh, node_map, boundaries = generate_2d_grid(N)
    dirichlet = np.array(boundaries["dirichlet"])

    psi = poisson_2d(S, dirichlet, mesh, node_map)

    assert np.allclose(psi, exact_sol(mesh), atol=1e-3)


def test_hard():
    """Test the FEM for a difficult source term."""
    N = 25
    S = (
        lambda x: 2
        * x[..., 0]
        * (x[..., 0] - 2)
        * (3 * x[..., 1] ** 2 - 3 * x[..., 1] + 0.5)
        + x[..., 1] ** 2 * (x[..., 1] - 1) ** 2
    )
    exact_sol = (
        lambda x: x[..., 0]
        * (1 - x[..., 0] / 2)
        * x[..., 1] ** 2
        * (1 - x[..., 1]) ** 2
    )

    mesh, node_map, boundaries = generate_2d_grid(N)
    dirichlet = np.array(boundaries["dirichlet"])

    psi = poisson_2d(S, dirichlet, mesh, node_map)

    assert np.allclose(psi, exact_sol(mesh), atol=1e-3)
