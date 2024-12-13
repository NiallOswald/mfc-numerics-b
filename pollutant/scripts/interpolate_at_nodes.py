#! /usr/bin/env python

from pollutant.utils import load_mesh
from argparse import ArgumentParser
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

MESHES = ["esw", "las"]
SCALES = {
    "esw": ["100k", "50k", "25k", "12_5k", "6_25k"],
    "las": ["40k", "20k", "10k", "5k", "2_5k", "1_25k"],
}


def interpolate_on_mesh(mesh, scale, path):
    """Interpolate the weather data at the nodes of the mesh."""
    # Load mesh
    try:
        nodes, _, _ = load_mesh(mesh, scale)
    except FileNotFoundError:
        print(f"Mesh {mesh} with scale {scale} not found. Skipping.")
        return

    # Create directory for the interpolated data
    if not os.path.exists(path.parent / f"{mesh}_{scale}"):
        os.makedirs(path.parent / f"{mesh}_{scale}")

    df = pd.read_csv(path)
    time_groups = df.groupby("ob_end_time")

    # Iterate over time values
    for datetime, group in time_groups:
        # Interpolate the weather data at the nodes
        u_velocity = griddata(
            (group["station_x"], group["station_y"]),
            group["horizontal_wind_speed"],
            (nodes[:, 0], nodes[:, 1]),
            method="linear",
            fill_value=0.0,
        )
        v_velocity = griddata(
            (group["station_x"], group["station_y"]),
            group["vertical_wind_speed"],
            (nodes[:, 0], nodes[:, 1]),
            method="linear",
            fill_value=0.0,
        )

        interpolated_data = pd.DataFrame(
            {"u_velocity": u_velocity, "v_velocity": v_velocity}
        )

        datetime_str = datetime.replace(" ", "_").replace(":", "_")

        interpolated_data.to_csv(
            path.parent / f"{mesh}_{scale}/{datetime_str}.csv", index=False
        )


def interpolate_at_nodes():
    parser = ArgumentParser()
    parser.add_help = """Interpolate the weather data at the nodes."""
    parser.add_argument("path", type=str, help="Path to the weather data.")
    parser.add_argument(
        "--mesh", choices=["esw", "las"], default="las", help="The mesh to use."
    )
    parser.add_argument(
        "--scale", type=str, default="10k", help="The scale of the mesh."
    )
    parser.add_argument(
        "--all", action="store_true", help="Interpolate all all meshes."
    )
    args = parser.parse_args()

    path = Path(args.path)

    if args.all:
        for mesh in MESHES:
            for scale in SCALES[mesh]:
                interpolate_on_mesh(mesh, scale, path)

    else:
        interpolate_on_mesh(args.mesh, args.scale, path)
