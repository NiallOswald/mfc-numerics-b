"""Solve the advection-diffusion equation on the provided mesh."""

from pollutant.finite_elements import LagrangeElement
from pollutant.reference_elements import ReferenceTriangle, ReferenceInterval
from pollutant.quadrature import gauss_quadrature
from pollutant.utils import load_mesh, find_element, gaussian_source
from pollutant.constants import (
    SOUTHAMPTON,
    READING,
    BURN_TIME,
    WIND_SPEED,
    DIFFUSION_RATE,
)

from alive_progress import alive_it
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import matplotlib.pyplot as plt


def dt_advection_diffusion(
    c,
    S,
    u,
    kappa,
    mesh,
    node_map,
    boundaries,
    boundary_type="Robin",
    return_norms=False,
    title="",
):
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

    # Setup empty list for the boundary normal vectors
    if return_norms:
        norms = []

    # Compute the global mass and stiffness matrix
    M = sp.lil_matrix((len(mesh), len(mesh)))
    K = sp.lil_matrix((len(mesh), len(mesh)))
    f = np.zeros(len(mesh))
    for nodes in alive_it(node_map, title=title + "Construcing stiffness matrix..."):
        J = np.einsum("ja,jb", mesh[nodes], fe.cell_jacobian, optimize=True)
        inv_J = np.linalg.inv(J)
        det_J = np.linalg.det(J)

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

        if boundary_type == "Robin":
            continue

        edges = np.column_stack([nodes, np.roll(nodes, -1)])
        for edge_index, edge in enumerate(edges):
            if not np.isin(edge, boundaries).all():
                continue

            reference_normal = ReferenceTriangle.cell_normals[edge_index]

            J_1d = np.einsum("ja,jb", mesh[edge], fe_1d.cell_jacobian).T[0]
            det_J_1d = np.linalg.norm(J_1d)

            normal = np.einsum(
                "ab,b->a",
                inv_J.T,
                reference_normal,
            )
            normal /= np.linalg.norm(normal)

            if return_norms:
                norms.append([mesh[edge], normal])

            if boundary_type == "Dirichlet":
                raise NotImplementedError(
                    "Dirichlet boundary conditions not implemented"
                )

            elif boundary_type == "Neumann":
                K[np.ix_(edge, edge)] = (
                    np.einsum(
                        "qa,qb,qc,ck,k,q->ab",
                        phi_1d,
                        phi_1d,
                        phi_1d,
                        u[edge],
                        normal,
                        weights_1d,
                    )
                ) * det_J_1d
            else:
                raise ValueError("Invalid boundary type")

    # Solve the system
    M = sp.csr_matrix(M)
    K = sp.csr_matrix(K)

    try:
        if c == "optimize":
            c_dt = lambda c: sp.linalg.spsolve(M, f - K @ c)
        else:
            c_dt = sp.linalg.spsolve(M, f - K @ c)
    except ValueError:
        c_dt = sp.linalg.spsolve(M, f - K @ c)

    if return_norms:
        return c_dt, norms
    else:
        return c_dt


def solve_advection_diffusion(
    t_final,
    kappa,
    source=SOUTHAMPTON,
    mesh="esw",
    scale="25k",
    max_step=1.0,
):
    # Load the mesh
    nodes, node_map, boundary_nodes = load_mesh(mesh, scale)

    # Set the parameters
    S = np.zeros(len(nodes))

    u = np.zeros_like(nodes)
    u[:, 1] = WIND_SPEED

    args = (S, u, kappa, nodes, node_map, boundary_nodes)

    # Cache the time derivatives
    coarse_times = np.linspace(0, BURN_TIME, 20)
    amplitudes = 1e-4 * gaussian_source(
        coarse_times[:, np.newaxis], BURN_TIME / 2.0, radius=BURN_TIME / 2.0
    )

    print("Caching time derivatives...")
    c_dts = []
    for i, (t, amplitude) in enumerate(zip(coarse_times, amplitudes)):
        S[:] = gaussian_source(
            nodes,
            source,
            amplitude=amplitude,
            radius=10000.0,
        )
        c_dts.append(
            dt_advection_diffusion(
                "optimize", *args, title=f"({i + 1} of {len(coarse_times)}) "
            )
        )

    # Solve the advection-diffusion equation
    c = np.zeros(len(nodes))
    sol = solve_ivp(
        lambda t, x: c_dts[coarse_times.searchsorted(t, "right") - 1](x),
        (0, t_final),
        c,
        method="LSODA",
        max_step=max_step,
    )

    return sol, nodes, node_map


def eval_target_concentration(sol, nodes, node_map, target=READING):
    """Evaluate the concentration at the target point."""
    # Locate the target element
    target_element = find_element(target, nodes, node_map)

    # Compute the concentration at the target point
    target_element_concentration = sol.y[node_map[target_element]]

    # Interpolate the concentration at the target point
    cg1 = LagrangeElement(ReferenceTriangle, 1)
    J = np.einsum(
        "ja,jb", nodes[node_map[target_element]], cg1.cell_jacobian, optimize=True
    )
    local_coords = np.linalg.solve(J, target - nodes[node_map[target_element][0]])
    target_nodes = cg1.tabulate([local_coords])[0]
    target_concentration = np.einsum(
        "a,at->t", target_nodes, target_element_concentration
    )

    return target_concentration


if __name__ == "__main__":
    # Set global parameters
    kappa = DIFFUSION_RATE
    t_final = 2 * BURN_TIME

    # Solve the advection-diffusion equation
    sol, nodes, node_map = solve_advection_diffusion(
        t_final, kappa, mesh="las", scale="10k", max_step=1e3
    )

    # Compute the concentration at the target point
    target_concentration = eval_target_concentration(sol, nodes, node_map)

    # Plot the concentration at the target point
    plt.plot(sol.t, target_concentration)
    plt.plot([BURN_TIME, BURN_TIME], [0, target_concentration.max()], "r--")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title("Concentration at Reading over time")
    plt.show()

    # Plot the concentration over the mesh
    for i in range(0, len(sol.t), len(sol.t) // 10):
        plt.tripcolor(
            nodes[:, 0],
            nodes[:, 1],
            node_map,
            sol.y[:, i],  # shading="gouraud"
        )
        plt.plot(*SOUTHAMPTON, "ro", label="Southampton")
        plt.plot(*READING, "bo", label="Reading")
        plt.colorbar()
        plt.title(f"Time: {sol.t[i]:.2f}")
        plt.show()
