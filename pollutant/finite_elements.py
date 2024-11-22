import numpy as np


def lagrange_points(degree):
    """Construct the locations of the Lagrange points for polynomials of the
    specified degree in two dimensions.
    """

    cube = np.indices([degree + 1] * 2)[::-1]
    coords_sum = np.sum(cube, axis=0)
    return np.stack(cube, axis=-1)[coords_sum <= degree] / degree


def vandermonde_matrix(degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree in two dimensions.
    """

    # Cast points to a np.ndarray
    points = np.array(points)

    # Construct the matrix of powers present in the Vandermonde matrix
    x_p = np.concatenate([np.arange(d, -1, -1) for d in range(degree + 1)])
    y_p = np.concatenate([np.arange(d + 1) for d in range(degree + 1)])

    i_p = np.vstack([x_p, y_p])

    if grad:
        # Modify powers of the coordinates to account for the derivatives
        d_p = i_p[:, np.newaxis, :] - np.eye(2)[..., np.newaxis]
        # Repeat grid points into a new axis
        point_mat = np.repeat(points[..., np.newaxis], 2, axis=2)
        # 'Outer-product'-like tensor power to compute all elements
        vand_grad = np.prod(point_mat[..., np.newaxis] ** d_p, axis=1)
        # Multiply by the powers of the coordinates to complete the derivatives
        vand_grad = np.einsum("ikj,kj->ijk", vand_grad, i_p, optimize=True)
        # Tidy up any NaNs or Infs
        vand_grad = np.nan_to_num(vand_grad, nan=0, posinf=0, neginf=0)

        return vand_grad

    # 'Outer-product'-like tensor power to compute all elements
    return np.prod(points[:, :, np.newaxis] ** i_p, axis=1)


class FiniteElement:
    def __init__(self, degree, nodes):
        self.degree = degree
        self.nodes = nodes

        # Compute the coefficients of the basis functions
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(degree, nodes))

    def tabulate(self, points, grad=False):
        """Tabulate the basis functions at the specified points."""
        return np.einsum(
            "ib...,bj->ij...",
            vandermonde_matrix(self.degree, points, grad),
            self.basis_coefs,
            optimize=True,
        )


class LagrangeElement(FiniteElement):
    def __init__(self, degree):
        self.degree = degree
        self.nodes = lagrange_points(degree)

        self._cell_jacobian = None

        super(LagrangeElement, self).__init__(degree, self.nodes)

    @property
    def cell_jacobian(self):
        if self._cell_jacobian is None:
            cg1 = LagrangeElement(1)
            self._cell_jacobian = cg1.tabulate(np.zeros((1, 2)), grad=True)[0]
        return self._cell_jacobian
