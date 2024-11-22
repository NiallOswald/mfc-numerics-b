from numpy.polynomial.legendre import leggauss
import numpy as np


def gauss_quadrature(degree, dim=2):
    """Compute the Gauss quadrature points and weights."""

    if dim == 1:
        npoints = int((degree + 1 + 1) / 2)

        points, weights = leggauss(npoints)

        points = (points + 1.0) / 2.0
        weights = weights / 2.0

    elif dim == 2:
        p1, w1 = gauss_quadrature(degree + 1, dim=1)
        p2, w2 = gauss_quadrature(degree, dim=1)

        points = np.array([(p, q * (1 - p)) for p in p1 for q in p2])
        weights = np.array([p * q * (1 - x) for p, x in zip(w1, p1) for q in w2])

    else:
        raise ValueError("Unknown cell dimension.")

    return points, weights
