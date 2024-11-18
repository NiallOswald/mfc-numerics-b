"""Exercise 3.1."""

import numpy as np
import matplotlib.pyplot as plt


def simple_fem(alpha, beta):
    K = np.array(
        [
            [1, 0, 0],
            [-2, 4, -2],
            [0, -2, 2],
        ]
    )
    f = np.array([alpha, 1 / 4, 1 / 24 + beta])

    # Solve the system
    c = np.linalg.solve(K, f)

    return c


def __main__():
    mesh = np.linspace(0, 1, 3)
    xx = np.linspace(0, 1, 100)
    exact_sol = lambda x: x * (x**2 - 3 * x + 3) / 6

    plt.plot(mesh, simple_fem(0, 0), "k-", label="Numerical")
    plt.plot(xx, exact_sol(xx), "k--", label="Exact")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\psi(x)$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
