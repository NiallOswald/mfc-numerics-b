"""Solve the advection-diffusion equation on the provided mesh."""

from pollutant import LagrangeElement, gauss_quadrature, load_mesh
from alive_progress import alive_it
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def dt_advection_diffusion(c, S, u, kappa, mesh, node_map, boundaries):
    """Compute the time-derivative for the advection-diffusion equation on a mesh."""
    # TODO: Implement the advection term
    assert np.allclose(u, 0.0), "Advection term is not supported"

    # Get the quadrature points and weights
    points, weights = gauss_quadrature(2)

    # Define the finite element
    fe = LagrangeElement(1)

    # Tabulate the shape functions and their gradients
    phi = fe.tabulate(points)
    grad_phi = fe.tabulate(points, grad=True)

    # Compute the global mass and stiffness matrix
    M = sp.lil_matrix((len(mesh), len(mesh)))
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros(len(mesh))
    for nodes in alive_it(node_map, title="Construcing stiffness matrix..."):
        J = np.einsum("ja,jb", mesh[nodes], fe.cell_jacobian, optimize=True)
        inv_J = np.linalg.inv(J)
        det_J = abs(np.linalg.det(J))

        M[np.ix_(nodes, nodes)] += (
            np.einsum(
                "qa,qb,q->ab",
                phi,
                phi,
                weights,
                optimize=True,
            )
        ) * det_J

        K[np.ix_(nodes, nodes)] += (
            -np.einsum(
                "qa,qci,ij,cj,qb,q->ab",
                phi,
                grad_phi,
                inv_J,
                u[nodes],
                phi,
                weights,
                optimize=True,
            )
            - np.einsum(
                "qai,ij,cj,qc,qb,q->ab",
                grad_phi,
                inv_J,
                u[nodes],
                phi,
                phi,
                weights,
                optimize=True,
            )
            + np.einsum(
                "qai,ik,qbj,jk,q->ab",
                grad_phi,
                inv_J,
                grad_phi,
                inv_J,
                weights,
                optimize=True,
            )
            * kappa
        ) * det_J

        f[nodes] += (
            np.einsum("qa,qb,b,q->a", phi, phi, S[nodes], weights, optimize=True)
            * det_J
        )

    # Set the dirichlet boundary conditions
    M[boundaries], K[boundaries], f[boundaries] = 0, 0, 0
    M[boundaries, boundaries] = 1

    # Solve the system
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)

    if c == "optimize":
        return lambda c: sp.linalg.spsolve(M, f - K @ c)
    else:
        return sp.linalg.spsolve(M, f - K @ c)


if __name__ == "__main__":
    source = np.array([450000, 175000])
    extent = 10000

    # Load the mesh
    nodes, node_map, boundary_nodes = load_mesh("esw", "25k")

    # Define the source term
    S = np.zeros(len(nodes))
    S[np.linalg.norm(nodes - source, axis=1) < extent] = 1.0

    u = np.zeros_like(nodes)
    kappa = 1e5

    t_final = 10000.0
    dt = 1e1

    args = (S, u, kappa, nodes, node_map, boundary_nodes)

    # Solve the diffusion equation
    c = np.zeros(len(nodes))
    c_dt = dt_advection_diffusion("optimize", *args)
    for i in alive_it(range(int(t_final / dt)), title="Iterating over time..."):
        c += dt * c_dt(c)

    plt.tripcolor(nodes[:, 0], nodes[:, 1], node_map, c)
    plt.colorbar()
    plt.show()
