"""Utilities for the finite element method."""

from pollutant.constants import FIRE_START
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.optimize import linprog

np.seterr(invalid="ignore", divide="ignore", over="ignore")


def load_mesh(name: str, scale: str, path: Path = Path("mesh")):
    """Load the mesh from a file."""
    nodes = np.loadtxt(path / f"{name}_grids/{name}_nodes_{scale}.txt")
    node_map = np.loadtxt(path / f"{name}_grids/{name}_IEN_{scale}.txt", dtype=np.int64)
    boundary_nodes = np.loadtxt(
        path / f"{name}_grids/{name}_bdry_{scale}.txt", dtype=np.int64
    )

    return nodes, node_map, boundary_nodes


def load_weather_data(
    name: str, scale: str, path: Path = Path("data"), init_time: datetime = FIRE_START
):
    """Load the weather data from a file."""
    wind_data = dict()
    file_paths = path.glob(f"{name}_{scale}/*.csv")
    for data_path in file_paths:
        time = datetime.strptime(Path(data_path).stem, "%Y-%m-%d_%H_%M_%S")
        time_in_seconds = (time - init_time).total_seconds()
        wind_data[time_in_seconds] = np.loadtxt(data_path, delimiter=",", skiprows=1)

    return wind_data


def save_gif(in_dir: Path, out_path: Path = Path("animation.gif"), **kwargs):
    """Create a gif from the images in a directory."""
    frames = [Image.open(image) for image in sorted(in_dir.glob("*.jpg"))]
    frame_one = frames[0]
    frame_one.save(
        out_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        **kwargs,
    )


def gaussian_source(x, x0, amplitude=1.0, radius=1.0, order=2.0):
    """A gaussian source term."""
    val = (
        amplitude
        * np.e
        * np.exp(-1 / (1 - np.linalg.norm((x - x0) / radius, axis=1) ** order))
    )
    val[np.linalg.norm(x - x0, axis=1) > radius] = 0.0
    return np.nan_to_num(val, nan=0.0)


def gaussian_source_simple(x, x0, amplitude=1.0, radius=1.0, order=2.0):
    if np.linalg.norm(x - x0) >= radius:
        return 0.0
    else:
        val = amplitude * np.exp(-1 / (1 - np.linalg.norm((x - x0) / radius) ** order))
        return np.nan_to_num(val, nan=0.0)


def find_element(x, nodes, node_map):
    """Find the element containing the point x."""
    for i, element in enumerate(node_map):
        if is_inside(x, nodes[element]):
            return i
    else:
        raise ValueError("Point not found in any element.")


def is_inside(x, points):
    """Check if the point x is inside the convex hull of the points."""
    A_eq = np.vstack([points.T, np.ones(len(points))])
    b_eq = np.array([*x, 1])
    cost = np.zeros(len(points))
    return linprog(cost, A_eq=A_eq, b_eq=b_eq).success
