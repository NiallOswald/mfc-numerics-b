"""Tests for exercise 3.1."""

from exercises.exercise_3_1 import simple_fem
import numpy as np


def test_simple():
    """Test the simple FEM implementation."""
    exact_sol = lambda x: x * (x**2 - 3 * x + 3) / 6

    mesh = np.linspace(0, 1, 3)
    psi = simple_fem(0, 0)

    assert np.allclose(psi, exact_sol(mesh))
