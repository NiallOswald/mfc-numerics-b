"""Tests for exercise 4.1."""

from exercises.exercise_4_1 import generate_2d_grid, J, J_inv, K_loc, f_loc, poisson_2d
import numpy as np

trimesh = np.array([[0, 0], [1, 0], [0, 1]])
trinode_map = np.array([[0, 1, 2]])


def test_J():
    assert np.allclose(J(*trimesh[trinode_map[0]]), np.eye(2))


def test_J_inv():
    assert np.allclose(J_inv(*trimesh[trinode_map[0]]), np.eye(2))


def test_K_loc():
    K = K_loc(*trimesh[trinode_map[0]])

    assert np.allclose(K, np.array([[1, -0.5, -0.5], [-0.5, 0.5, 0], [-0.5, 0, 0.5]]))


def test_f_loc():
    S = lambda x: 1.0
    f = f_loc(S, *trimesh[trinode_map[0]])

    assert np.allclose(f, np.array([1 / 6, 1 / 6, 1 / 6]))


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
