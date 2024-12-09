#! /usr/bin/env python
from alive_progress import alive_it
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path

YEAR = "2005"
PATTERN = f"**/qc-version-1/*{YEAR}.csv"
META_PATTERN = "**/*station-metadata.csv"
DATA_HEADER = 80  # Metadata rows to skip
META_HEADER = 48


def knots_to_mps(knots):
    """Convert knots to meters per second."""
    return knots * 0.514444


knots_to_mps = np.vectorize(knots_to_mps)


def proc_windspeed(df):
    """Convert wind speed and direction to horizontal and vertical components."""
    df["mean_wind_speed"] = knots_to_mps(df["mean_wind_speed"])
    df["mean_wind_dir"] = np.radians(df["mean_wind_dir"])
    df["horizontal_wind_speed"] = df["mean_wind_speed"] * np.sin(df["mean_wind_dir"])
    df["vertical_wind_speed"] = df["mean_wind_speed"] * np.cos(df["mean_wind_dir"])

    df.drop(columns=["mean_wind_dir", "mean_wind_speed"], inplace=True)


def proc_lat_lon(df, path):
    """Add the latitude and longitude of the weather stations."""
    meta_path = path.glob(META_PATTERN)

    with open(list(meta_path)[0], "r") as csvfile:
        meta_df = pd.read_csv(csvfile, header=META_HEADER)

    meta_df = meta_df.iloc[:-1]  # Drop the last row
    meta_df["src_id"] = meta_df["src_id"].astype(int)

    df = df.merge(
        meta_df[["src_id", "station_latitude", "station_longitude"]],
        on="src_id",
        how="left",
    )

    return df[
        [
            "ob_end_time",
            "station_latitude",
            "station_longitude",
            "horizontal_wind_speed",
            "vertical_wind_speed",
        ]
    ]


def process_weather_data():
    parser = ArgumentParser()
    parser.add_help = """Process MetOffice MIDAS Open: UK mean wind data.

    Data is available at: https://dx.doi.org/10.5285/91cb9985a6c2453d99084bde4ff5f314.
    """
    parser.add_argument("path", type=str, help="Path to the data.")
    parser.add_argument(
        "--start",
        type=str,
        default="2005-10-30 06:00:00",
        help="Start of datetime range to include.",
    )
    parser.add_argument(
        "--end", type=str, default="2005-10-31 06:00:00", help="End of datetime range."
    )
    parser.add_argument(
        "--output", type=str, default="wind_data.csv", help="Output path for the CSV."
    )
    args = parser.parse_args()

    path = Path(args.path)
    start = args.start
    end = args.end
    output = args.output

    pathlist = path.glob(PATTERN)

    dataframes = []
    for path in alive_it(pathlist, title="Searching for files..."):
        with open(path, "r") as csvfile:
            df = pd.read_csv(
                csvfile,
                header=DATA_HEADER,
                index_col=0,
            )
            df = df[start:end]

            dataframes.append(df)

    df = pd.concat(dataframes)

    # Post processing
    df.reset_index(inplace=True)
    df["src_id"] = df["src_id"].astype(int)

    df = df[
        [
            "ob_end_time",
            "src_id",
            "mean_wind_dir",
            "mean_wind_speed",
        ]
    ]
    df.dropna(inplace=True)

    proc_windspeed(df)
    df = proc_lat_lon(df, Path(args.path))

    df.to_csv(output)
