from .reference_elements import ReferenceInterval, ReferenceTriangle
from numpy.polynomial.legendre import leggauss
import numpy as np


def gauss_quadrature(cell, degree):
    """Compute the Gauss quadrature points and weights."""

    if cell is ReferenceInterval:
        npoints = int((degree + 1 + 1) / 2)

        points, weights = leggauss(npoints)

        points = (points + 1.0) / 2.0
        weights = weights / 2.0

        points.shape = [points.shape[0], 1]

    elif cell is ReferenceTriangle:
        p1, w1 = gauss_quadrature(ReferenceInterval, degree + 1)
        p2, w2 = gauss_quadrature(ReferenceInterval, degree)

        points = np.array([(p[0], q[0] * (1 - p[0])) for p in p1 for q in p2])
        weights = np.array([p * q * (1 - x[0]) for p, x in zip(w1, p1) for q in w2])

    else:
        raise ValueError("Unknown reference cell")

    return points, weights
