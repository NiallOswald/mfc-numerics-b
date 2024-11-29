import numpy as np


class ReferenceCell:
    def __init__(self, vertices):
        self.vertices = vertices
        self.dim = self.vertices.shape[1]


ReferenceInterval = ReferenceCell(np.array([[0.0], [1.0]]))
ReferenceTriangle = ReferenceCell(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
