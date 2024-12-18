# mfc-numerics-b
Exercises and final report for the Part B of the MFC CDT Numerical Analysis course

## Installation
The package is installable using `pip` by the following command:
```
> python -m pip install .
```

> Note: It is recommended to install Python packages within a virutal environment or use a package manager like Conda.

## Downloading Windspeed Data
For reasons of copyright protection, wind speed data is not included as part of this repository. However a script, `download_data.sh`, is included to download the data from CEDA and process it appropriately. This script requires an API key which you will receive upon creating an account. Note that the script will proceed to download a few thousand files totaling to approximately 8GB of data. Depending on your device this may take up to 20 minutes to complete. Once the data is downloaded, the processing is relatively fast.

If you have downloaded the data through other means, the following scripts can be run (in order) to process the data:
```
> process_weather_data /path/to/data
> transform_station_data /path/to/data
> interpolate_at_nodes /path/to/data --all
```

If no data is found, the solver will automatically default to a wind speed of $10 \: ms^{-1}$ due North.

## Scripts
The advection diffusion solver can be run with the command:
```
> python pollutant/solvers/advection_diffusion
```
> **WARNING:** `AdvectionDiffusion.save_animation` will save figures to a temporary directory (defaults to `./tmp`). After saving the animation as a GIF it will delete all files with the extension `*.jpg` within the temporary directory. To protect data, the program will terminate if it detects that the temporary directory is not empty.

After the script has completed a full-length run on a mesh of your choosing, you will be prompted whether to run a convergence test. Do note that this is very time-consuming, however the convergence test may be interrupted at any time with `Ctrl+C` without loss of data.
