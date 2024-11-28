"""Exercise 4.1."""

from .utils import generate_2d_grid
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


gauss_points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])


def global_map(xi, x_0, x_1, x_2):
    """Map the local coordinates to the global coordinates."""
    return np.einsum("i,ij->j", phi(xi), np.array([x_0, x_1, x_2]))


def phi(xi):
    """Compute the shape functions."""
    return np.array([1 - xi[0] - xi[1], xi[0], xi[1]])


def grad_phi():
    """Compute the gradient of the shape functions."""
    return np.array([[-1, -1], [1, 0], [0, 1]])


def J(x_0, x_1, x_2):
    """Compute the Jacobian."""
    return np.array(
        [[x_1[0] - x_0[0], x_2[0] - x_0[0]], [x_1[1] - x_0[1], x_2[1] - x_0[1]]]
    )


def J_inv(x_0, x_1, x_2):
    """Compute the inverse of the Jacobian."""
    return np.linalg.inv(J(x_0, x_1, x_2))


def K_loc(*e):
    """Compute the local stiffness matrix."""
    return (
        np.einsum(
            "ai,ij,bk,kj->ab",
            grad_phi(),
            J_inv(*e),
            grad_phi(),
            J_inv(*e),
        )
        * abs(np.linalg.det(J(*e)))
        / 2
    )


def f_loc(S, *e):
    """Compute the local forcing vector."""
    return (
        np.sum(
            [phi(xi) * S(global_map(xi, *e)) for xi in gauss_points],
            axis=0,
        )
        * abs(np.linalg.det(J(*e)))
        / 6
    )


def poisson_2d(S, dirichlet, mesh, node_map):
    # Compute the global stiffness matrix
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros(len(mesh))
    for e in node_map:
        K[np.ix_(e, e)] += K_loc(*mesh[e])
        f[e] += f_loc(S, *mesh[e])

    # Set the dirichlet boundary conditions
    K[dirichlet] = 0
    K[dirichlet, dirichlet] = 1
    f[dirichlet] = 0

    # Solve the system
    K = sp.csr_matrix(K)
    c = sp.linalg.spsolve(K, f)

    return c


def __main__():
    # Parameters
    N_ELEMS = 10
    S = (
        lambda x: 2
        * x[..., 0]
        * (x[..., 0] - 2)
        * (3 * x[..., 1] ** 2 - 3 * x[..., 1] + 0.5)
        + x[..., 1] ** 2 * (x[..., 1] - 1) ** 2
    )

    # Generate a mesh
    mesh, node_map, boundaries = generate_2d_grid(N_ELEMS)

    dirichlet = np.array(boundaries["dirichlet"])

    # Solve the system
    psi = poisson_2d(S, dirichlet, mesh, node_map)

    # Plot the solution
    plt.tripcolor(mesh[:, 0], mesh[:, 1], node_map, psi)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.colorbar()
    plt.show()

    # Plot the solution as a surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(mesh[:, 0], mesh[:, 1], psi)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$\psi$")
    plt.show()


if __name__ == "__main__":
    __main__()
