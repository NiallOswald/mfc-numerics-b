from .reference_elements import ReferenceInterval, ReferenceTriangle
import numpy as np

np.seterr(invalid="ignore", divide="ignore")


def lagrange_points(cell, degree):
    """Construct the locations of the Lagrange points for polynomials of the
    specified degree in two dimensions.
    """

    cube = np.indices([degree + 1] * cell.dim)[::-1]
    coords_sum = np.sum(cube, axis=0)
    return np.stack(cube, axis=-1)[coords_sum <= degree] / degree


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree in two dimensions.
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
    def __init__(self, cell, degree, nodes):
        self.cell = cell
        self.degree = degree
        self.nodes = nodes

        # Compute the coefficients of the basis functions
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

    def tabulate(self, points, grad=False):
        """Tabulate the basis functions at the specified points."""
        return np.einsum(
            "ib...,bj->ij...",
            vandermonde_matrix(self.cell, self.degree, points, grad),
            self.basis_coefs,
            optimize=True,
        )

    def interpolate(self, fn):
        """Interpolate the specified function onto the finite element."""
        return np.array([fn(node) for node in self.nodes])


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        nodes = lagrange_points(cell, degree)

        self._cell_jacobian = None

        super(LagrangeElement, self).__init__(cell, degree, nodes)

    @property
    def cell_jacobian(self):
        if self._cell_jacobian is None:
            cg1 = LagrangeElement(self.cell, 1)
            self._cell_jacobian = cg1.tabulate(np.zeros((1, self.cell.dim)), grad=True)[
                0
            ]
        return self._cell_jacobian
