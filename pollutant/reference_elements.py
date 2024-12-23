"""Reference elements for finite elements."""

import numpy as np


class ReferenceCell:
    """A reference cell."""

    def __init__(self, vertices: np.ndarray, cell_normals: np.ndarray):
        """Initialise the reference cell.

        :param vertices: An array of shape (n, d) containing the vertices of the cell.
        :param cell_normals: An array of shape (n, d) containing the normals to the
            cell facets.
        """
        self.vertices = vertices
        self.cell_normals = cell_normals
        self.dim = self.vertices.shape[1]


ReferenceInterval = ReferenceCell(np.array([[0.0], [1.0]]), None)
ReferenceTriangle = ReferenceCell(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, -1.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [-1.0, 0.0]]),
)
