# import libs
import argparse
import os
import pathlib

import numpy as np
import xarray as xr


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path, help="Path to folder containing files")
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Output path, include the file name with .zarr ending",
    )
    return parser.parse_args()


def merge_zarr_files(zarr_path, merged_zarr_path):
    # Collect paths of Zarr files in the specified directory
    zarr_files = [
        os.path.join(zarr_path, file) for file in os.listdir(zarr_path) if file.endswith(".zarr")
    ]

    # Open the datasets and store them in a list
    datasets = [xr.open_dataset(file) for file in zarr_files]

    # Concatenate the datasets along the 'init_time' dimension
    merged_ds = xr.concat(datasets, dim="init_time")

    merged_ds = merged_ds.sortby("init_time")

    # Define the specific range of x and y coordinates
    # x_range = (-10, 2)  # Example x coordinate range
    # y_range = (49, 59)  # Example y coordinate range

    # Iterate over the remaining Zarr files and merge them into the initial dataset
    # for file in zarr_files[1:]:
    #     xr.open_zarr(file)
    #     print(file)

    #     # ds_filt = ds.sel(x=slice(*x_range), y=slice(*y_range))
    #     merged_ds = merged_ds.combine_first(ds_filt)

    # Rechunk the merged dataset
    # merged_ds = merged_ds.chunk(chunks={"init_time": 10, "x": 100, "y": 100})

    # Get dims/coords into correct type
    step_hours = merged_ds["step"].values
    step_timedelta = np.timedelta64(1, "h") * step_hours
    ds_timedelta = merged_ds.assign_coords(step=step_timedelta)

    # Save the merged dataset as a new Zarr file
    ds_timedelta.to_zarr(merged_zarr_path)


def main():
    args = _parse_args()
    merge_zarr_files(args.input, args.output)


# Check if script is being run directly
if __name__ == "__main__":
    main()
