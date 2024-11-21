"""Exercise 4.1."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


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


gauss_points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])


def global_map(xi, x_0, x_1, x_2):
    return np.einsum("i,ij->j", phi(xi), np.array([x_0, x_1, x_2]))


def phi(xi):
    return np.array([1 - xi[0] - xi[1], xi[0], xi[1]])


def grad_phi():
    return np.array([[-1, -1], [1, 0], [0, 1]])


def J(x_0, x_1, x_2):
    return np.array(
        [[x_1[0] - x_0[0], x_2[0] - x_0[0]], [x_1[1] - x_0[1], x_2[1] - x_0[1]]]
    )


def J_inv(x_0, x_1, x_2):
    return np.linalg.inv(J(x_0, x_1, x_2))


def poisson_2d(S, dirichlet, mesh, node_map):
    # Compute the local stiffness matrix
    K_loc = lambda *e: np.einsum(
        "ai,ij,bk,kj->ab",
        grad_phi(),
        J_inv(*e),
        grad_phi(),
        J_inv(*e),
    ) * np.linalg.det(J(*e))
    f_loc = (
        lambda *e: np.sum(
            [
                phi(gauss_points[i]) * S(global_map(gauss_points[i], *e))
                for i in range(3)
            ],
            axis=0,
        )
        * np.linalg.det(J(*e))
        / 3  # TODO: this should be 6
    )

    # Compute the global stiffness matrix
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros(len(mesh))
    for e in node_map:
        K[np.ix_(e, e)] += K_loc(*mesh[e])
        f[e] += f_loc(*mesh[e])

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


if __name__ == "__main__":
    __main__()
