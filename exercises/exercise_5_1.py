"""Exercise 5.1."""

from alive_progress import alive_bar
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


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


def dt_advection_diffusion_1d(psi, S, u, mu, beta, mesh, node_map):
    # Compute the global stiffness and mass matrix
    M = sp.lil_matrix((len(mesh), len(mesh)))
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros_like(mesh)
    for e in node_map:
        M[np.ix_(e, e)] += M_loc(*mesh[e])
        K[np.ix_(e, e)] += K_loc(*mesh[e], mu, u)
        f[e] += f_loc(*mesh[e], S)

    dirichlet = 0
    neumann = -1

    # Set the dirichlet boundary condition
    M[dirichlet], K[dirichlet], f[dirichlet] = 0, 0, 0
    M[dirichlet, dirichlet] = 1

    # Set the neumann boundary conditions
    e = node_map[neumann]
    K[np.ix_(e, e)] += u * phi(1)[:, np.newaxis] @ phi(1)[np.newaxis, :]
    f[neumann] += mu * beta

    # Solve the system
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)

    if psi == "optimize":
        return lambda psi: sp.linalg.spsolve(M, f - K @ psi)
    else:
        return sp.linalg.spsolve(M, f - K @ psi)


def __main__():
    # Parameters
    N_ELEMS = 20
    psi_init = lambda x: 0.0
    S = lambda x: 1.0 - x
    exact = lambda x: x * (x**2 - 3 * x + 3) / 6

    u = 0.0
    mu = 1.0
    beta = 0

    dt = 1e-4
    t_final = 1.0

    n_steps = int(t_final / dt)

    # Generate a uniform mesh
    mesh = np.linspace(0, 1, N_ELEMS + 1)

    psi = np.array([psi_init(x) for x in mesh])

    # Setup the node map
    node_order = np.arange(N_ELEMS, dtype=np.int64)[:, np.newaxis]
    node_map = np.hstack([node_order, node_order + 1])

    # Cache the time derivative
    args = (S, u, mu, beta, mesh, node_map)
    psi_dt = dt_advection_diffusion_1d("optimize", *args)

    # Solve the system using the method of lines
    with alive_bar(n_steps) as bar:
        for i in range(n_steps):
            psi += dt * psi_dt(psi)
            bar()

    # Plot the solution
    plt.plot(mesh, psi, "k-", label=rf"Numerical $t = {t_final}$")
    plt.plot(mesh, exact(mesh), "k--", label="Exact")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x, t)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
