# mfc-numerics-b
Exercises and final report for the Part B of the MFC CDT Numerical Analysis course

## Installation

## Downloading Windspeed Data
For reasons of copyright protection, wind speed data is not included as part of this repository. However a script `download_data.sh` is included to download the data from CEDA and process it appropriately. This script requires an API key which you will receive upon creating an account. Note that the script will proceed to download a few thousand files totaling to approximately 8GB of data. Depending on your device this may take up to 20 minutes to complete. Once the data is downloaded, the processing is relatively fast.

If you have downloaded the data through other means, the following scripts can be run (in order) to process the data:
1) `process_weather_data /path/to/data`
2) `transform_station_data /path/to/data`
3) `interpolate_at_nodes /path/to/data --all`

If no data is found, the solver will automatically default to a wind speed of 10ms-1 due North.
