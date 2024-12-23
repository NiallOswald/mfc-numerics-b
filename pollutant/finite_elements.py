"""Finite elements on reference cells."""

from .reference_elements import ReferenceCell, ReferenceInterval, ReferenceTriangle
from typing import Callable
import numpy as np

np.seterr(invalid="ignore", divide="ignore")


def lagrange_points(cell: ReferenceCell, degree: int) -> np.ndarray:
    """Construct the locations of the Lagrange points for polynomials of the
    specified degree in two dimensions.

    Adapted from an implementation of `fe_utils
        <https://github.com/Imperial-MATH60022/finite-element-2022-NiallOswald>`.

    :param cell: The :class:.`~reference_elements.ReferenceCell` to use
    :param degree: The degree of the polynomials.

    :returns: An array of shape (n, 2) containing the coordinates of the
        Lagrange points.
    """

    cube = np.indices([degree + 1] * cell.dim)[::-1]
    coords_sum = np.sum(cube, axis=0)
    return np.stack(cube, axis=-1)[coords_sum <= degree] / degree


def vandermonde_matrix(
    cell: ReferenceCell, degree: int, points: list, grad: bool = False
) -> np.ndarray:
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree in two dimensions.

    Adapted from an implementation of `fe_utils
        <https://github.com/Imperial-MATH60022/finite-element-2022-NiallOswald>`.

    :param cell: The :class:.`~reference_elements.ReferenceCell` to use.
    :param degree: The degree of the polynomials.
    :param points: An array of shape (m, 2) containing the coordinates of the points
        at which to evaluate the Vandermonde matrix.
    :param grad: If True, return the gradient of the Vandermonde matrix.

    :returns: If grad is False, an array of shape (m, n) containing the Vandermonde
        matrix. If grad is True, an array of shape (m, n, cell.dim)
    """

    # Cast points to a np.ndarray
    points = np.array(points)

    # Construct the matrix of powers present in the Vandermonde matrix
    if cell is ReferenceInterval:
        i_p = np.arange(degree + 1).reshape(1, degree + 1)

    elif cell is ReferenceTriangle:
        x_p = np.concatenate([np.arange(d, -1, -1) for d in range(degree + 1)])
        y_p = np.concatenate([np.arange(d + 1) for d in range(degree + 1)])

        i_p = np.vstack([x_p, y_p])

    else:
        raise ValueError("Unknown reference cell")

    if grad:
        # Modify powers of the coordinates to account for the derivatives
        d_p = i_p[:, np.newaxis, :] - np.eye(cell.dim)[:, :, np.newaxis]
        # Repeat grid points into a new axis
        point_mat = np.repeat(points[:, :, np.newaxis], cell.dim, axis=2)
        # 'Outer-product'-like tensor power to compute all elements
        vand_grad = np.prod(point_mat[:, :, :, np.newaxis] ** d_p, axis=1)
        # Multiply by the powers of the coordinates to complete the derivatives
        vand_grad = np.einsum("ikj,kj->ijk", vand_grad, i_p, optimize=True)
        # Tidy up any NaNs or Infs
        vand_grad = np.nan_to_num(vand_grad, nan=0, posinf=0, neginf=0)

        return vand_grad

    # 'Outer-product'-like tensor power to compute all elements
    return np.prod(points[:, :, np.newaxis] ** i_p, axis=1)


class FiniteElement:
    """A finite element on a reference cell."""

    def __init__(self, cell: ReferenceCell, degree: int, nodes: np.ndarray):
        """Initialise the finite element.

        :param cell: The :class:.`~reference_elements.ReferenceCell` of the finite
            element.
        :param degree: The degree of the finite element.
        :param nodes: An array of shape (n, 2) containing the coordinates of the nodes
            of the finite element.
        """
        self.cell = cell
        self.degree = degree
        self.nodes = nodes

        # Compute the coefficients of the basis functions
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

    def tabulate(self, points: np.ndarray, grad: bool = False) -> np.ndarray:
        """Tabulate the basis functions at the specified points.

        Adapted from an implementation of `fe_utils
            <https://github.com/Imperial-MATH60022/finite-element-2022-NiallOswald>`.

        :param points: An array of shape (m, 2) containing the coordinates of the
            points at which to evaluate the basis functions.
        :param grad: If True, return the gradient of the basis functions.

        :returns: If grad is False, an array of shape (m, n) containing the
            basis functions. If grad is True, an array of shape (m, n, cell.dim)
            containing the gradients of the basis functions.
        """
        return np.einsum(
            "ib...,bj->ij...",
            vandermonde_matrix(self.cell, self.degree, points, grad),
            self.basis_coefs,
            optimize=True,
        )

    def interpolate(self, fn: Callable) -> np.ndarray:
        """Interpolate the specified function onto the nodes of a finite element.

        :param fn: A function that takes a point and returns a scalar value.

        :returns: An array of shape (n,) containing the basis function coefficients.
        """
        return np.array([fn(node) for node in self.nodes])


class LagrangeElement(FiniteElement):
    """A Lagrange finite element on a reference cell."""

    def __init__(self, cell: ReferenceCell, degree: int):
        """Initialise the finite element.

        :param cell: The :class:.`~reference_elements.ReferenceCell` of the finite
            element.
        :param degree: The degree of the finite element.
        """
        nodes = lagrange_points(cell, degree)

        self._cell_jacobian = None

        super(LagrangeElement, self).__init__(cell, degree, nodes)

    @property
    def cell_jacobian(self) -> np.ndarray:
        """The Jacobian of the finite element.

        Adapted from an implementation of `fe_utils
            <https://github.com/Imperial-MATH60022/finite-element-2022-NiallOswald>`.

        :returns: An array of shape (cell.dim, cell.dim) containing the Jacobian.
        """
        if self._cell_jacobian is None:
            cg1 = LagrangeElement(self.cell, 1)
            self._cell_jacobian = cg1.tabulate(np.zeros((1, self.cell.dim)), grad=True)[
                0
            ]
        return self._cell_jacobian
