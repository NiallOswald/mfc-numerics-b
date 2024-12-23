"""Utilities for the pollutant model."""

from pollutant.constants import FIRE_START
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.optimize import linprog

np.seterr(invalid="ignore", divide="ignore", over="ignore")


def load_mesh(
    name: str, scale: str, path: Path = Path("mesh")
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the mesh from a file.

    :param name: The name of the mesh.
    :param scale: The scale of the mesh.
    :param path: The path to the mesh files.

    :returns: A tuple containing the nodes, node map, and boundary nodes.
    """
    nodes = np.loadtxt(path / f"{name}_grids/{name}_nodes_{scale}.txt")
    node_map = np.loadtxt(path / f"{name}_grids/{name}_IEN_{scale}.txt", dtype=np.int64)
    boundary_nodes = np.loadtxt(
        path / f"{name}_grids/{name}_bdry_{scale}.txt", dtype=np.int64
    )

    return nodes, node_map, boundary_nodes


def load_weather_data(
    name: str, scale: str, path: Path = Path("data"), init_time: datetime = FIRE_START
) -> dict[float, np.ndarray]:
    """Load the weather data from a file.

    :param name: The name of the mesh.
    :param scale: The scale of the mesh.
    :param path: The path to the weather data files.
    :param init_time: The initial datetime of the simulation.

    :returns: A dictionary containing the weather data at each available time step with
        keys taken as the time in seconds from `init_time`.
    """
    wind_data = dict()
    file_paths = path.glob(f"{name}_{scale}/*.csv")
    for data_path in file_paths:
        time = datetime.strptime(Path(data_path).stem, "%Y-%m-%d_%H_%M_%S")
        time_in_seconds = (time - init_time).total_seconds()
        wind_data[time_in_seconds] = np.loadtxt(data_path, delimiter=",", skiprows=1)

    return wind_data


def save_gif(im_dir: Path, out_path: Path = Path("animation.gif"), **kwargs) -> None:
    """Create and save a gif from images in a directory.

    :param im_dir: The directory containing the images.
    :param out_path: The path to save the gif to.
    :param kwargs: Additional arguments to pass to `PIL.Image.save`.
    """
    frames = [Image.open(image) for image in sorted(im_dir.glob("*.jpg"))]
    frame_one = frames[0]
    frame_one.save(
        out_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        **kwargs,
    )


def gaussian_source(
    x: np.ndarray,
    x0: np.ndarray,
    amplitude: float = 1.0,
    radius: float = 1.0,
    order: float = 2.0,
) -> np.ndarray:
    """A smooth gaussian-like bump function.

    :param x: The points at which to evaluate the source.
    :param x0: The centre of the source.
    :param amplitude: The amplitude of the source.
    :param radius: The radius of the support.
    :param order: The rate of decay of the source.

    :returns: An array of shape (m,) containing the source values at each point.
    """
    val = (
        amplitude
        * np.e
        * np.exp(-1 / (1 - np.linalg.norm((x - x0) / radius, axis=1) ** order))
    )
    val[np.linalg.norm(x - x0, axis=1) > radius] = 0.0
    return np.nan_to_num(val, nan=0.0)


def gaussian_source_simple(
    x: np.ndarray,
    x0: np.ndarray,
    amplitude: float = 1.0,
    radius: float = 1.0,
    order: float = 2.0,
) -> float:
    """A smooth gaussian-like bump function.

    :param x: The point at which to evaluate the source.
    :param x0: The centre of the source.
    :param amplitude: The amplitude of the source.
    :param radius: The radius of the support.
    :param order: The rate of decay of the source.

    :returns: The source value at the point.
    """
    if np.linalg.norm(x - x0) >= radius:
        return 0.0
    else:
        val = amplitude * np.exp(-1 / (1 - np.linalg.norm((x - x0) / radius) ** order))
        return np.nan_to_num(val, nan=0.0)


def find_element(x: np.ndarray, nodes: np.ndarray, node_map: np.ndarray) -> int:
    """Find the element containing the point of specified corrdinates.

    :param x: The coordinates of the point to locate.
    :param nodes: The coordinates of the nodes of the mesh.
    :param node_map: The map of elements to nodes.

    :returns: The index of the element containing the point.
    """
    for i, element in enumerate(node_map):
        if is_inside(x, nodes[element]):
            return i
    else:
        raise ValueError("Point not found in any element.")


def is_inside(x: np.ndarray, points: np.ndarray) -> bool:
    """Check if the point x is inside the convex hull of the points.

    :param x: The point to check.
    :param points: The points defining the convex hull.

    :returns: True if the point is inside the convex hull, False otherwise.
    """
    A_eq = np.vstack([points.T, np.ones(len(points))])
    b_eq = np.array([*x, 1])
    cost = np.zeros(len(points))
    return linprog(cost, A_eq=A_eq, b_eq=b_eq).success
