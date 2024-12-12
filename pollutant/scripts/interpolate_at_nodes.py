#! /usr/bin/env python

from pollutant.utils import load_mesh
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata


def interpolate_at_nodes():
    parser = ArgumentParser()
    parser.add_help = """Interpolate the weather data at the nodes."""
    parser.add_argument("path", type=str, help="Path to the weather data.")
    parser.add_argument("mesh", choices=["esw", "las"], help="The mesh to use.")
    parser.add_argument("scale", type=str, help="The scale of the mesh.")
    args = parser.parse_args()

    path = Path(args.path)

    nodes, _, _ = load_mesh(args.mesh, args.scale)
    df = pd.read_csv(path)

    # Interpolate the weather data at the nodes
    u_velocity = griddata(
        (df["station_x"], df["station_y"]),
        df["horizontal_wind_speed"],
        (nodes[:, 0], nodes[:, 1]),
        method="linear",
        fill_value=0.0,
    )
    v_velocity = griddata(
        (df["station_x"], df["station_y"]),
        df["vertical_wind_speed"],
        (nodes[:, 0], nodes[:, 1]),
        method="linear",
        fill_value=0.0,
    )

    interpolated_data = pd.DataFrame(
        {"u_velocity": u_velocity, "v_velocity": v_velocity}
    )
    interpolated_data.to_csv(
        path.with_name(path.stem + "_interpolated.csv"), index=False
    )
