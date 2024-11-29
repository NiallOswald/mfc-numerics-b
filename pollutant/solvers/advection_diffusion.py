"""Solve the advection-diffusion equation on the provided mesh."""

from pollutant import (
    LagrangeElement,
    ReferenceTriangle,
    ReferenceInterval,
    gauss_quadrature,
    load_mesh,
)
from alive_progress import alive_it
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def dt_advection_diffusion(c, S, u, kappa, mesh, node_map, boundaries):
    """Compute the time-derivative for the advection-diffusion equation on a mesh."""
    # Get the quadrature points and weights
    points, weights = gauss_quadrature(ReferenceTriangle, 2)
    points_1d, weights_1d = gauss_quadrature(ReferenceInterval, 2)

    # Define the finite element
    fe = LagrangeElement(ReferenceTriangle, 1)
    fe_1d = LagrangeElement(ReferenceInterval, 1)

    # Tabulate the shape functions and their gradients
    phi = fe.tabulate(points)
    grad_phi = fe.tabulate(points, grad=True)

    phi_1d = fe_1d.tabulate(points_1d)

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

        boundary_nodes = np.array([node for node in nodes if node in boundaries])

        if len(boundary_nodes) == 2:
            J_1d = np.einsum("ja,jb", mesh[boundary_nodes], fe_1d.cell_jacobian)
            det_J_1d = np.linalg.norm(J_1d)
            normal = np.array([J_1d[1, 0], -J_1d[0, 0]]) / det_J_1d

            K[np.ix_(boundary_nodes, boundary_nodes)] += (
                np.einsum(
                    "qa,qb,qc,ck,k,q->ab",
                    phi_1d,
                    phi_1d,
                    phi_1d,
                    u[boundary_nodes],
                    normal,
                    weights_1d,
                )
            ) * det_J_1d

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
    nodes, node_map, boundary_nodes = load_mesh("esw", "12_5k")

    # Define the source term
    S = np.zeros(len(nodes))
    S[np.linalg.norm(nodes - source, axis=1) < extent] = 1e-3

    u = np.zeros_like(nodes)
    u[:, 1] = 50
    kappa = 1e2

    t_final = 1200
    dt = 1e-2

    args = (S, u, kappa, nodes, node_map, boundary_nodes)

    # Solve the diffusion equation
    c = np.zeros(len(nodes))
    c_dt = dt_advection_diffusion("optimize", *args)
    for i in alive_it(range(int(t_final / dt)), title="Iterating over time..."):
        c += dt * c_dt(c)

    plt.tripcolor(nodes[:, 0], nodes[:, 1], node_map, c)
    plt.colorbar()
    plt.show()
