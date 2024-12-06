"""Solve the diffusion equation on the provided mesh."""

from pollutant.finite_elements import LagrangeElement
from pollutant.reference_elements import ReferenceTriangle
from pollutant.quadrature import gauss_quadrature
from pollutant.utils import load_mesh, find_element, gaussian_source
from pollutant.constants import SOUTHAMPTON, READING

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def solve_diffusion(S, mesh, node_map, boundaries):
    """Solve the diffusion equation on a mesh."""
    # Get the quadrature points and weights
    points, weights = gauss_quadrature(ReferenceTriangle, 2)

    # Define the finite element
    fe = LagrangeElement(ReferenceTriangle, 1)

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
            np.einsum("qi,k,qk,q->i", phi, S[nodes], phi, weights, optimize=True)
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
    # Load the mesh
    nodes, node_map, boundary_nodes = load_mesh("esw", "6_25k")

    # Define the source term
    S = gaussian_source(nodes, SOUTHAMPTON, amplitude=1e-7, radius=10000.0, order=2.0)

    # Solve the diffusion equation
    c = solve_diffusion(S, nodes, node_map, boundary_nodes)

    # Locate the target element
    target_element = find_element(READING, nodes, node_map)

    # Compute the concentration at the target point
    target_element_concentration = c[node_map[target_element]]

    # Interpolate the concentration at the target point
    cg1 = LagrangeElement(ReferenceTriangle, 1)
    J = np.einsum(
        "ja,jb", nodes[node_map[target_element]], cg1.cell_jacobian, optimize=True
    )
    local_coords = np.linalg.solve(J, READING - nodes[node_map[target_element][0]])
    target_nodes = cg1.tabulate([local_coords])[0]
    target_concentration = np.dot(target_nodes, target_element_concentration)

    print(f"Concentration at Reading: {target_concentration}")

    # Plot the concentration over the mesh
    plt.tripcolor(nodes[:, 0], nodes[:, 1], node_map, c)
    plt.plot(*SOUTHAMPTON, "ro", label="Southampton")
    plt.plot(*READING, "bo", label="Reading")
    plt.colorbar()
    plt.show()
