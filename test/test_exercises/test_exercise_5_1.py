"""Tests for exercise 5.1."""

from exercises.exercise_3_2 import poisson_1d
from exercises.exercise_5_1 import dt_advection_diffusion_1d
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("t_final, tol", zip([0.5, 1.0, 2.0], [1e-2, 1e-2, 1e-2]))
def test_steady_state(t_final, tol, N_ELEMS=20, dt=1e-4):
    # Parameters
    S = lambda x: 1.0 - x
    u = 0.0
    mu = 1.0
    alpha = 0.0
    beta = 0.0

    t_final = 1.0

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, N_ELEMS + 1)

    # Setup the node map
    node_order = np.arange(N_ELEMS, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Solve the system using the method of lines
    psi = np.zeros_like(mesh)
    args = (S, u, mu, alpha, beta, mesh, node_map)
    for i in range(int(t_final / dt)):
        psi_step = psi + dt * dt_advection_diffusion_1d(psi, *args)
        psi = (psi + psi_step + dt * dt_advection_diffusion_1d(psi_step, *args)) / 2

    plt.plot(mesh, psi, "k-", label=r"$\psi(x, t)$")
    plt.plot(
        mesh, poisson_1d(S, alpha, beta, mesh, node_map), "k--", label=r"$\psi(x)$"
    )
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x, t)$")
    plt.legend()
    plt.show()

    # Check the solution
    assert np.allclose(psi, poisson_1d(S, alpha, beta, mesh, node_map), atol=tol)
