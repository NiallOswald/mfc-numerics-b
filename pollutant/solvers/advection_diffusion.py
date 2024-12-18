"""Solve the advection-diffusion equation on the provided mesh."""

from pollutant.finite_elements import LagrangeElement
from pollutant.reference_elements import ReferenceTriangle
from pollutant.quadrature import gauss_quadrature
import pollutant.utils as utils
from pollutant.constants import (
    SOUTHAMPTON,
    READING,
    BURN_TIME,
    DEFAULT_WIND_SPEED,
    DIFFUSION_RATE,
)

from alive_progress import alive_it, alive_bar
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.integrate import solve_ivp
import scipy.sparse as sp

plt.rcParams["font.size"] = 12

ESW_SCALES = ["100k", "50k", "25k", "12_5k", "6_25k"]
LAS_SCALES = ["40k", "20k", "10k", "5k", "2_5k", "1_25k"]


class AdvectionDiffusion:
    VECTORIZED = {
        "RK45": False,
        "RK23": False,
        "Radau": False,
        "BDF": True,
        "LSODA": False,
    }
    CALL_ESTIMATES = {
        "RK45": 5,  # Accurate
        "RK23": 3,  # Accurate
        "Radau": 5,  # Inaccurate
        "BDF": 1,  # Accurate
        "LSODA": 5,  # Inaccurate
    }

    def __init__(
        self,
        kappa,
        mesh,
        scale,
        boundary_type="Robin",
        data_path=Path("data"),
    ):
        self.kappa = kappa
        self.mesh = mesh
        self.scale = scale
        self.boundary_type = boundary_type
        self.data_path = data_path

        self.nodes, self.node_map, self.boundary_nodes = utils.load_mesh(mesh, scale)
        self.node_count = len(self.nodes)

        self.S = utils.gaussian_source(
            self.nodes, SOUTHAMPTON, radius=10000.0, order=2.0
        )
        print("Auto-nomalizing the source...")
        self.norm = self.auto_normalize()
        print("Done!")

        self._cache_assembly()

    @staticmethod
    def _amplitude(t):
        return utils.gaussian_source_simple(t, BURN_TIME / 2.0, radius=BURN_TIME / 2.0)

    def amplitude(self, t):
        return self.norm * self._amplitude(t)

    def auto_normalize(self):
        """Automatically normalize the source."""
        # Integrate the source over the domain
        # Get the quadrature points and weights
        points, weights = gauss_quadrature(ReferenceTriangle, 2)

        # Define the finite element
        fe = LagrangeElement(ReferenceTriangle, 1)

        # Tabulate the shape functions and their gradients
        phi = fe.tabulate(points)

        # Compute the global mass and stiffness matrix
        total_mass = 0.0
        for e in alive_it(self.node_map, title="Integrating the source..."):
            J = np.einsum("ja,jb", self.nodes[e], fe.cell_jacobian, optimize=True)
            det_J = abs(np.linalg.det(J))

            total_mass += (
                np.einsum(
                    "qa,a,q",
                    phi,
                    self.S[e],
                    weights,
                    optimize=True,
                )
                * det_J
            )

        # Integrate the amplitude over time
        time_grid = np.linspace(0, 2 * BURN_TIME, 100)
        amplitude_vals = np.array([self._amplitude(t) for t in time_grid])
        total_amplitude = np.trapezoid(amplitude_vals, time_grid)

        return 1 / (total_amplitude * total_mass)

    def load_weather_data(self):
        """Load the weather data."""
        try:
            u_data = utils.load_weather_data(self.mesh, self.scale, self.data_path)

        except FileNotFoundError:
            print("No weather data found. Using constant default wind speed.")
            u = np.zeros_like(self.nodes)
            u[:, :] = DEFAULT_WIND_SPEED
            u_data = {np.inf: u}

        return u_data

    def _cache_assembly(self):
        """Cache the stiffness and mass matrices."""
        print("Caching stiffness and mass matrices...")
        u_data = self.load_weather_data()
        self.t_data = np.sort(np.array(list(u_data.keys())))
        self.matrix_data = [
            self._assemble(
                self.S, u_data[t], title=f"({(i + 1):02} / {len(self.t_data)}) "
            )
            for i, t in enumerate(self.t_data)
        ]

        # Sort the data
        self.matrix_ = np.argsort(self.t_data)

    def _assemble(self, S, u, title=""):
        """Assemble the mass and stiffness matrices."""
        # Get the quadrature points and weights
        points, weights = gauss_quadrature(ReferenceTriangle, 2)

        # Define the finite element
        fe = LagrangeElement(ReferenceTriangle, 1)

        # Tabulate the shape functions and their gradients
        phi = fe.tabulate(points)
        grad_phi = fe.tabulate(points, grad=True)

        # Compute the global mass and stiffness matrix
        M = sp.lil_matrix((self.node_count, self.node_count))
        K = sp.lil_matrix((self.node_count, self.node_count))
        f = np.zeros(self.node_count)
        for e in alive_it(
            self.node_map, title=title + "Construcing stiffness matrix..."
        ):
            J = np.einsum("ja,jb", self.nodes[e], fe.cell_jacobian, optimize=True)
            inv_J = np.linalg.inv(J)
            det_J = abs(np.linalg.det(J))

            M[np.ix_(e, e)] += (
                np.einsum(
                    "qa,qb,q->ab",
                    phi,
                    phi,
                    weights,
                    optimize=True,
                )
            ) * det_J

            if self.boundary_type == "Robin":
                K[np.ix_(e, e)] += (
                    -np.einsum(
                        "qc,cj,qai,ij,qb,q->ab",
                        phi,
                        u[e],
                        grad_phi,
                        inv_J,
                        phi,
                        weights,
                        optimize=True,
                    )  # -(u . grad(phi)) * c
                    + np.einsum(
                        "qai,ik,qbj,jk,q->ab",
                        grad_phi,
                        inv_J,
                        grad_phi,
                        inv_J,
                        weights,
                        optimize=True,
                    )  # grad(phi) . grad(c)
                    * self.kappa
                ) * det_J

            elif self.boundary_type == "Neumann":
                K[np.ix_(e, e)] += (
                    np.einsum(
                        "qa,qb,ci,qcj,ji,q->ab",
                        phi,
                        phi,
                        u[e],
                        grad_phi,
                        inv_J,
                        weights,
                        optimize=True,
                    )  # phi * c * div(u)
                    + np.einsum(
                        "qa,qc,ci,qbj,ji,q->ab",
                        phi,
                        phi,
                        u[e],
                        grad_phi,
                        inv_J,
                        weights,
                        optimize=True,
                    )  # phi * (u . grad(c))
                    + np.einsum(
                        "qai,ik,qbj,jk,q->ab",
                        grad_phi,
                        inv_J,
                        grad_phi,
                        inv_J,
                        weights,
                        optimize=True,
                    )  # grad(phi) . grad(c)
                    * self.kappa
                ) * det_J

            f[e] += (
                np.einsum("qa,qb,b,q->a", phi, phi, S[e], weights, optimize=True)
                * det_J
            )  # phi * S

        if self.boundary_type == "Dirichlet":
            M[self.boundary_nodes, :] = 0.0
            M[self.boundary_nodes, self.boundary_nodes] = 1.0
            K[self.boundary_nodes, :] = 0.0
            f[self.boundary_nodes] = 0.0

        # Solve the system
        M = sp.csr_matrix(M)
        K = sp.csr_matrix(K)

        return M, K, f

    def assemble(self, t):
        M, K, f = self.matrix_data[self.t_data.searchsorted(t, "right") - 1]
        return M, K, self.amplitude(t) * f

    def _step(self, t, c):
        M, K, f = self.assemble(t)
        return sp.linalg.spsolve(M, f - K @ c)

    def _step_vectorized(self, t, c):
        M, K, f = self.assemble(t)
        b_mat = np.zeros(2 * [M.shape[0]])
        b_mat[:, : c.shape[1]] = c

        c_dt_mat = sp.linalg.spsolve(M, f - K @ b_mat)

        return c_dt_mat[:, : c.shape[1]]

    def step(self, t, c, vectorized=False, bar=lambda: None):
        """Compute the time-derivative."""
        bar()  # Update the progress bar

        if vectorized:
            return self._step_vectorized(t, c)
        else:
            return self._step(t, c)

    def solve(self, t_final, max_step, t_eval=None, method="RK23", bar_length=None):
        """Solve the advection-diffusion equation."""
        if bar_length is None:
            bar_length = self.CALL_ESTIMATES[method] * int(t_final / max_step)

        with alive_bar(
            bar_length, title="Solving advection-diffusion equation..."
        ) as bar:  # Progress is estimated using the number of calls
            c = np.zeros(self.node_count)
            sol = solve_ivp(
                lambda t, x: self.step(
                    t, x, vectorized=self.VECTORIZED[method], bar=bar
                ),
                (0, t_final),
                c,
                method=method,
                max_step=max_step,
                t_eval=t_eval,
                vectorized=self.VECTORIZED[method],
            )

        self.sol = sol

        if sol.status != 0:
            print("Warning: Integration failed.")

        return sol

    def eval_target_concentration(self, target):
        """Evaluate the concentration at the target point."""
        # Locate the target element
        target_element = utils.find_element(target, self.nodes, self.node_map)

        # Compute the concentration at the target point
        target_element_concentration = self.sol.y[self.node_map[target_element]]

        # Interpolate the concentration at the target point
        cg1 = LagrangeElement(ReferenceTriangle, 1)
        J = np.einsum(
            "ja,jb",
            self.nodes[self.node_map[target_element]],
            cg1.cell_jacobian,
            optimize=True,
        )
        local_coords = np.linalg.solve(
            J, target - self.nodes[self.node_map[target_element][0]]
        )
        target_nodes = cg1.tabulate([local_coords])[0]
        target_concentration = np.einsum(
            "a,at->t", target_nodes, target_element_concentration
        )

        return target_concentration

    def integrate_target_concentration(self, target):
        """Integrate the concentration at the target point."""
        # Compute the concentration at the target point
        target_concentration = self.eval_target_concentration(target)

        # Integrate the concentration at the target point
        return np.trapezoid(target_concentration, self.sol.t)

    def plot_target_concentration(self, target, savefig=False):
        # Compute the concentration at the target point
        target_concentration = self.eval_target_concentration(target)

        # Plot the concentration at the target point
        plt.figure(figsize=(8, 6))
        plt.plot(self.sol.t, target_concentration, "k-")
        plt.plot([BURN_TIME, BURN_TIME], [0, target_concentration.max()], "k--")
        plt.xlabel(r"Time ($s$)")
        plt.ylabel(r"Concentration ($m^{-2}$)")
        plt.title("Concentration at Reading over time")

        if savefig:
            plt.savefig("target_concentration.pdf", dpi=300)
        else:
            plt.show()

    def eval_total_concentration(self):
        # Get the quadrature points and weights
        points, weights = gauss_quadrature(ReferenceTriangle, 2)

        # Define the finite element
        fe = LagrangeElement(ReferenceTriangle, 1)

        # Tabulate the shape functions
        phi = fe.tabulate(points)

        # Compute the total concentration at each time
        total_concentration = np.zeros(len(self.sol.t))
        for e in alive_it(self.node_map, title="Evaluting total concentration..."):
            J = np.einsum("ja,jb", self.nodes[e], fe.cell_jacobian, optimize=True)
            det_J = abs(np.linalg.det(J))

            total_concentration += (
                np.einsum(
                    "qa,at,q->t",
                    phi,
                    self.sol.y[e],
                    weights,
                    optimize=True,
                )
            ) * det_J

        return total_concentration

    def plot_total_concentration(self, savefig=False):
        # Compute the total concentration at each time
        total_concentration = self.eval_total_concentration()

        # Plot the total concentration
        plt.figure(figsize=(8, 6))
        plt.plot(self.sol.t, total_concentration, "k-")
        plt.plot([BURN_TIME, BURN_TIME], [0, total_concentration.max()], "k--")
        plt.xlabel(r"Time ($s$)")
        plt.ylabel("Pollutant")
        plt.title("Total mass of pollutant over time")

        if savefig:
            plt.savefig("total_concentration.pdf", dpi=300)
        else:
            plt.show()

    def save_animation(self, frames=None, temp_dir=Path("./tmp")):
        # Setup
        if frames is None:
            frames = range(len(self.sol.t))

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        elif os.listdir(temp_dir) != []:
            raise FileExistsError(
                "Temporary directory is not empty. Please clear it before continuing."
            )

        try:
            # Save the frames
            for i in alive_it(frames, total=len(frames), title="Saving figures..."):
                # Plot the concentration
                plt.figure(figsize=(8, 6))
                plt.tripcolor(
                    self.nodes[:, 0], self.nodes[:, 1], self.node_map, self.sol.y[:, i]
                )
                plt.plot(*SOUTHAMPTON, "ro", label="Southampton")
                plt.plot(*READING, "bo", label="Reading")

                plt.title(f"Concentration at time {self.sol.t[i]:.2f}")
                plt.legend()
                plt.colorbar()

                plt.tight_layout()

                # Save the figure
                plt.savefig(f"tmp/{i:04d}.jpg")

                # Close the figure
                plt.close()

            # Create the gif
            print("Creating gif...")
            utils.save_gif(temp_dir)

        except KeyboardInterrupt:
            pass

        finally:
            # Clean up
            print("Cleaning up...")
            for file in temp_dir.glob("*.jpg"):
                os.remove(file)  # This is dangerous


def compute_convergence(eval_time, func, kappa, scales, mesh, max_step=1e0):
    """Compute the convergence of the advection-diffusion equation."""
    # Iterate over the scales
    values = []
    for scale in scales:
        print(f"Running for {scale} on {mesh} mesh...")

        try:
            # Solve the advection-diffusion equation
            eq = AdvectionDiffusion(kappa, mesh, scale)
            sol = eq.solve(eval_time, max_step=max_step, t_eval=[eval_time])

            # Store the value
            if sol.status == 0:
                print(func(eq))
                values.append(func(eq))
            else:  # Store NaN if the integration failed
                values.append(np.nan)

        except KeyboardInterrupt:  # Allow for early termination
            print("Terminating early...")
            break

    return values


if __name__ == "__main__":
    # Set global parameters
    kappa = DIFFUSION_RATE
    t_final = 2.0 * BURN_TIME
    max_step = 1e1
    t_eval = np.linspace(0, t_final, 1000)

    # Solve the advection-diffusion equation
    eq = AdvectionDiffusion(kappa, mesh, scale)
    sol = eq.solve(t_final, max_step=max_step, t_eval=t_eval)

    # Plot the concentration at the target point
    eq.plot_target_concentration(READING, savefig=True)

    # Compute the integral of the concentration at the target point
    print(
        "Integral of concentration at Reading:",
        eq.integrate_target_concentration(READING),
    )

    # Plot the total concentration
    eq.plot_total_concentration(savefig=True)

    # Create an animation of the concentration over the mesh
    # I am aware FuncAnimation exists, but it is unbearably slow in this case
    print("Creating animation...")
    eq.save_animation(frames=range(0, len(sol.t), 10))
    print("Done!")

    # Compute the convergence
    eval_time = BURN_TIME
    # mesh, scales_str = "esw", ["100k", "50k", "25k", "12_5k", "6_25k"]
    mesh, scales_str = "las", ["40k", "20k", "10k", "5k", "2_5k"]  # , "1_25k"]
    scales = [1e3 * float(s.replace("_", ".").replace("k", "")) for s in scales_str]

    print("Computing convergence...")
    values = compute_convergence(
        eval_time,
        lambda eq: eq.eval_target_concentration(READING)[0],
        kappa,
        scales_str,
        mesh,
    )
    print("Done!")

    # Report the convergence
    print("Values:", values)
