"""Constants for the pollutant model."""

from datetime import datetime
import numpy as np

SOUTHAMPTON = np.array([442365.0, 115483.0])  # Easting, Northing
READING = np.array([473993.0, 171625.0])  # Easting, Northing

BURN_TIME = 10.0 * 60.0**2.0  # 10 hours in seconds
FIRE_START = datetime(2005, 10, 30, 6, 0, 0)  # 6am on 30th October 2005

DEFAULT_WIND_SPEED = [0.0, 10.0]  # m/s
DIFFUSION_RATE = 10000.0  # m^2/s
