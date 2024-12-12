#! /usr/bin/env python

from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from pyproj import Transformer


def transform_station_data():
    parser = ArgumentParser()
    parser.add_help = (
        """Transform the station coordinates from EPSG:4326 to EPSG:27700 (BNG)."""
    )
    parser.add_argument("path", type=str, help="Path to the data.")
    args = parser.parse_args()

    path = Path(args.path)

    with open(path, "r") as csvfile:
        df = pd.read_csv(csvfile)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

    df["station_x"], df["station_y"] = transformer.transform(
        df["station_longitude"].values, df["station_latitude"].values
    )
    df = df.drop(columns=["station_latitude", "station_longitude"])

    df.to_csv(path, index=False)
