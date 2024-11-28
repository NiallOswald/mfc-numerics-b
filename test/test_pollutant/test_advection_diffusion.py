"""Tests for the advection-diffusion solver."""

from pollutant.solvers.advection_diffusion import dt_advection_diffusion
from exercises.utils import generate_2d_grid
from pollutant.solvers.diffusion import solve_diffusion
import numpy as np
import pytest


@pytest.mark.parametrize("t_final, tol", zip([2.5, 4.0], [1e-3, 1e-4]))
def test_steady_state(t_final, tol):
    """Test the advection-diffusion solver for a steady state solution."""
    # Parameters
    N_ELEMS = 25
    dt = 1e-4

    S = lambda x: 1.0 - x[..., 0]
    u = lambda x: 0.0 * x
    kappa = 1.0

    # Generate a uniform mesh
    mesh, node_map, boundaries = generate_2d_grid(N_ELEMS)
    dirichlet = np.array(boundaries["dirichlet"])

    # Cache the time derivative
    psi_dt = dt_advection_diffusion(
        "optimize", S(mesh), u(mesh), kappa, mesh, node_map, dirichlet
    )

    # Solve the system using the method of lines
    psi = np.zeros(len(mesh))
    for i in range(int(t_final / dt)):
        psi += dt * psi_dt(psi)

    assert np.allclose(
        psi, solve_diffusion(S(mesh), mesh, node_map, dirichlet), atol=tol
    )
