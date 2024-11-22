"""Solve the diffusion equation on the provided mesh."""

from pollutant import LagrangeElement, gauss_quadrature, load_mesh
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def solve_diffusion(fn, mesh, node_map, boundaries):
    """Solve the diffusion equation on a mesh."""
    # Get the quadrature points and weights
    points, weights = gauss_quadrature(2)

    # Define the finite element
    fe = LagrangeElement(1)

    # Tabulate the shape functions and their gradients
    phi = fe.tabulate(points)
    grad_phi = fe.tabulate(points, grad=True)

    # Compute the global stiffness matrix
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros(len(mesh))
    for e, nodes in enumerate(node_map):
        J = np.einsum("ja,jb", mesh[nodes], fe.cell_jacobian, optimize=True)
        inv_J = np.linalg.inv(J)
        det_J = np.linalg.det(J)

        K[np.ix_(nodes, nodes)] += (
            np.einsum(
                "ba,qib,ya,qjy,q->ij",
                inv_J,
                grad_phi,
                inv_J,
                grad_phi,
                weights,
                optimize=True,
            )
            * det_J
        )

        f[nodes] += (
            np.einsum("qi,k,qk,q->i", phi, fn[nodes], phi, weights, optimize=True)
            * det_J
        )

    # Set the dirichlet boundary conditions
    K[boundaries] = 0
    K[boundaries, boundaries] = 1
    f[boundaries] = 0

    # Solve the system
    K = sp.csr_matrix(K)
    c = sp.linalg.spsolve(K, f)

    return c


if __name__ == "__main__":
    source = np.array([450000, 175000])
    extent = 10000

    # Load the mesh
    nodes, node_map, boundary_nodes = load_mesh("esw", "6_25k")

    # Define the source term
    fn = np.zeros(len(nodes))
    fn[np.linalg.norm(nodes - source, axis=1) < extent] = 1.0

    # Solve the diffusion equation
    c = solve_diffusion(fn, nodes, node_map, boundary_nodes)

    plt.tripcolor(nodes[:, 0], nodes[:, 1], node_map, c)
    plt.colorbar()
    plt.show()
