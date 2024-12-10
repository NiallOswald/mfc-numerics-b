#! /usr/bin/env python
from pollutant.utils import load_mesh
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_weather_stations():
    parser = ArgumentParser()
    parser.add_help = """Plot the weather stations on the mesh."""
    parser.add_argument("path", type=str, help="Path to the weather data.")
    parser.add_argument("mesh", choices=["esw", "las"], help="The mesh to use.")
    parser.add_argument("scale", type=str, help="The scale of the mesh.")
    args = parser.parse_args()

    path = Path(args.path)

    nodes, node_map, boundary_nodes = load_mesh(args.mesh, args.scale)
    df = pd.read_csv(path)

    plt.figure(figsize=(10, 10))
    plt.triplot(nodes[:, 0], nodes[:, 1], node_map)
    plt.scatter(df["station_x"], df["station_y"], c="r", label="Weather stations")
    plt.show()
