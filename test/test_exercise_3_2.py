"""Tests for exercise 3.2."""

from exercises.exercise_3_1 import simple_fem
from exercises.exercise_3_2 import poisson_1d
import numpy as np
import pytest


@pytest.mark.parametrize("alpha, beta", [(0, 0), (0.2, 0.5), (-0.4, 0.2), (1, 0)])
def test_simple(alpha, beta):
    """Test the FEM against the simple implementation."""
    n_elems = 2
    S = lambda x: 1 - x  # noqa: E731

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, n_elems + 1)

    # Setup the node map
    node_order = np.arange(n_elems, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Solve the system
    psi_new = poisson_1d(S, alpha, beta, mesh, node_map)
    psi_old = simple_fem(S, alpha, beta, mesh, node_map)

    # Check the solution
    assert np.allclose(psi_new, psi_old)


def test_convergence():
    """Test the convergence rate of the FEM."""
    n_vals = 2 ** np.arange(3, 12)
    S = lambda x: (1 - x) ** 2  # noqa: E731
    exact_sol = lambda x: x * (4 - 6 * x + 4 * x**2 - x**3) / 12  # noqa: E731

    # Compute the error
    errors = []
    for n in n_vals:
        # Generate a uniform mesh
        mesh = np.linspace(0, 1, n + 1)

        # Setup the node map
        node_order = np.arange(n, dtype=np.int64)[:, np.newaxis]
        node_map = np.hstack([node_order, node_order + 1])

        # Solve the system
        psi = poisson_1d(S, 0, 0, mesh, node_map)

        # Compute the error
        errors.append(np.linalg.norm(psi - exact_sol(mesh), np.inf))

    # Test the convergence rate
    a, _ = np.polyfit(np.log(n_vals), np.log(errors), 1)
    assert np.isclose(a, -2, atol=0.1)


def test_discontinuous():
    """Test the FEM against a discontinuous source term."""
    n_elems = 1000
    S = lambda x: float(abs(x - 0.5) < 0.25)

    def exact_sol(x):
        if x < 0.25:
            return 0.3 * x + 0.1
        elif 0.25 <= x < 0.75:
            return -0.5 * x**2 + 0.55 * x + 11 / 160
        else:
            return -0.2 * x + 0.35

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, n_elems + 1)

    # Setup the node map
    node_order = np.arange(n_elems, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Solve the system
    psi = poisson_1d(S, 0.1, -0.2, mesh, node_map)

    # Check the solution
    assert np.allclose(psi, np.array([exact_sol(x) for x in mesh]), atol=1e-3)
