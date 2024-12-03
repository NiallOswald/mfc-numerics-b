import numpy as np


class ReferenceCell:
    def __init__(self, vertices, cell_normals):
        self.vertices = vertices
        self.cell_normals = cell_normals
        self.dim = self.vertices.shape[1]


ReferenceInterval = ReferenceCell(np.array([[0.0], [1.0]]), None)
ReferenceTriangle = ReferenceCell(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, -1.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], [-1.0, 0.0]]),
)
