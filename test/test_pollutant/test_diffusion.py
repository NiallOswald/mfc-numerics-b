"""Tests for the diffusion solver."""

from pollutant.solvers.diffusion import solve_diffusion
from exercises.utils import generate_2d_grid
from exercises.exercise_4_1 import poisson_2d
import numpy as np
import pytest


def test_simple():
    """Test the FEM for a simple source term."""
    N = 25
    S = lambda x: 1.0 + 0 * x[..., 0]
    exact_sol = lambda x: x[..., 0] * (1 - x[..., 0] / 2)

    mesh, node_map, boundaries = generate_2d_grid(N)
    dirichlet = np.array(boundaries["dirichlet"])

    psi = solve_diffusion(S(mesh), mesh, node_map, dirichlet)

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

    psi = solve_diffusion(S(mesh), mesh, node_map, dirichlet)

    assert np.allclose(psi, exact_sol(mesh), atol=1e-3)


@pytest.mark.parametrize(
    "N, S",
    zip(
        [20, 25, 50],
        [
            lambda x: x[..., 0] ** 2 + x[..., 1] ** 2,
            lambda x: x[..., 0] * x[..., 1],
            lambda x: x[..., 0] + x[..., 1],
        ],
    ),
)
def test_equivalance(N, S):
    """Test that the diffusion solver is equivalent to the Poisson solver."""
    mesh, node_map, boundaries = generate_2d_grid(N)
    dirichlet = np.array(boundaries["dirichlet"])

    psi_diffusion = solve_diffusion(S(mesh), mesh, node_map, dirichlet)
    psi_poisson = poisson_2d(S, dirichlet, mesh, node_map)

    assert np.allclose(psi_diffusion, psi_poisson, atol=1e-3)
