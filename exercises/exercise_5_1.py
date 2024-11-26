"""Exercise 5.1."""

from alive_progress import alive_bar
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time


def global_map(xi, a, b):
    return a + (b - a) * xi


def phi(xi):
    return np.array([1 - xi, xi])


def grad_phi(xi):
    return np.array([-1.0, 1.0])


def J(a, b):
    return b - a


def inv_J(a, b):
    return 1 / J(a, b)


def M_loc(a, b):
    return np.array([[2, 1], [1, 2]]) * J(a, b) / 6


def K_loc(a, b, mu, u):
    return (
        -u * np.array([[-1, -1], [1, 1]]) * J(a, b) * inv_J(a, b) / 2
        + mu * np.array([[1, -1], [-1, 1]]) * J(a, b) * inv_J(a, b) ** 2
    )


def f_loc(a, b, S):
    return np.array([2 * S(a) + S(b), S(a) + 2 * S(b)]) * J(a, b) / 6


def dt_advection_diffusion_1d(psi, S, u, mu, alpha, beta, mesh, node_map):
    # Compute the global stiffness and mass matrix
    M = sp.lil_matrix((len(mesh), len(mesh)))
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros_like(mesh)
    for e in node_map:
        M[np.ix_(e, e)] += M_loc(*mesh[e])
        K[np.ix_(e, e)] += K_loc(*mesh[e], mu, u)
        f[e] += f_loc(*mesh[e], S)

    left = node_map[0]
    right = node_map[-1]

    # Set the dirichlet boundary conditions
    """ K[np.ix_(left, left)] += (
        mu * phi(0)[:, np.newaxis] @ grad_phi(0)[np.newaxis, :] * inv_J(*mesh[left])
    )
    print("AHHHHH")
    print(phi(0)[:, np.newaxis] @ grad_phi(0)[np.newaxis, :] * inv_J(*mesh[left]))
    f[0] += u * alpha """

    # Set the neumann boundary conditions
    K[np.ix_(right, right)] += u * phi(1)[:, np.newaxis] @ phi(1)[np.newaxis, :]
    f[-1] += mu * beta

    # Set the dirichlet boundary condition
    M[0] = 0
    K[0] = 0
    f[0] = 0
    M[0, 0] = 1

    # Solve the system
    time_start = time.time()
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)
    dtpsi = sp.linalg.spsolve(M, f - K @ psi)
    print(time.time() - time_start)

    return dtpsi


def __main__():
    # Parameters
    N_ELEMS = 10
    psi_init = lambda x: -(x - 0.4) * (x - 0.6) if abs(x - 0.5) < 0.1 else 0
    S = lambda x: 0.0
    u = 1.0
    mu = 0.1
    alpha = 0
    beta = 0

    dt = 1e-3
    n_steps = 100

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, N_ELEMS + 1)

    psi = np.array([psi_init(x) for x in mesh])

    # Setup the node map
    node_order = np.arange(N_ELEMS, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Solve the system using the method of lines
    args = (S, u, mu, alpha, beta, mesh, node_map)
    with alive_bar(n_steps) as bar:
        for i in range(n_steps):
            psi_step = psi + dt * dt_advection_diffusion_1d(psi, *args)
            bar()

    # Plot the solution
    plt.plot(mesh, psi, "k-", label=r"$\psi(x, t)$")
    plt.plot(mesh, np.array([psi_init(x) for x in mesh]), "k--", label=r"$\psi(x, 0)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x, t)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
