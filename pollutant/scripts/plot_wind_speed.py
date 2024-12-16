#! /usr/bin/env python
from pollutant.utils import load_mesh, load_weather_data
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pathlib import Path


def plot_wind_speed():
    parser = ArgumentParser()
    parser.add_help = """Plot the wind speed on the mesh."""
    parser.add_argument("path", type=str, help="Path to the weather data.")
    parser.add_argument("mesh", choices=["esw", "las"], help="The mesh to use.")
    parser.add_argument("scale", type=str, help="The scale of the mesh.")
    args = parser.parse_args()

    path = Path(args.path)

    nodes, _, _ = load_mesh(args.mesh, args.scale)
    wind_data = load_weather_data(args.mesh, args.scale, path)

    # Loop over the data and plot the wind speed
    fig, ax = plt.subplots(figsize=(5, 8) if args.mesh == "esw" else (8, 8))
    for time in sorted(wind_data.keys()):
        data = wind_data[time]
        ax.cla()
        ax.quiver(nodes[:, 0], nodes[:, 1], data[:, 0], data[:, 1])
        ax.set_title(f"Wind speed at {time} seconds")
        plt.pause(1)
