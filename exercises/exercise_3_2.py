"""Exercise 3.2."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def poisson_1d(S, alpha, beta, mesh, node_map):
    # Compute the local stiffness matrix
    K_loc = lambda a, b: np.array([[1, -1], [-1, 1]]) / (b - a)
    f_loc = lambda a, b: (b - a) / 6 * np.array([2 * S(a) + S(b), S(a) + 2 * S(b)])

    # Compute the global stiffness matrix
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros_like(mesh)
    for e in node_map:
        K[np.ix_(e, e)] += K_loc(*mesh[e])
        f[e] += f_loc(*mesh[e])

    # Set the dirichlet boundary conditions
    K[0] = 0
    K[0, 0] = 1
    f[0] = alpha

    # Set the neumann boundary conditions
    f[-1] += beta

    # Solve the system
    K = sp.csr_matrix(K)
    c = sp.linalg.spsolve(K, f)

    return c


def __main__():
    # Parameters
    N_ELEMS = 100
    S = lambda x: (1 - x) ** 2

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, N_ELEMS + 1)

    # Setup the node map
    node_order = np.arange(N_ELEMS, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Solve the system
    psi = poisson_1d(S, 0, 0, mesh, node_map)

    # Plot the solution
    plt.plot(mesh, psi, "k-")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x)$")
    plt.show()


if __name__ == "__main__":
    __main__()
