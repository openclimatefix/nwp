import argparse
import os
import pathlib
from datetime import datetime

import gcsfs
import numpy as np
import xarray as xr


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Output zarr file")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )

    return parser.parse_args()


def extract_files(args):
    # initialize a GCSFileSystem
    gcs = gcsfs.GCSFileSystem(project="excarta")
    path = f"gs://excarta-public-us/pilots/ocf/{args.year}/"
    files = gcs.ls(path)
    datasets = []

    for file in files:
        filename = os.path.basename(file)

        # extract date part from filename
        date_part = filename.split(".")[0]  # adjust this line if necessary

        # # convert date_part into a datetime
        date = datetime.strptime(date_part, "%Y%m%d%H")  # adjust format string if necessary

        print(date)
        # convert the date to numpy datetime64
        date_np = np.datetime64(date)

        # load the Zarr store as an xarray Dataset
        ds = xr.open_zarr(gcs.get_mapper(file), consolidated=True)
        ds = ds.assign_coords(ts=date_np)

        # calculate time differences in hours
        step_values = (ds["datetimes"].values - date_np) / np.timedelta64(1, "h")
        ds = ds.assign_coords(time=step_values)
        ds = ds.rename({"time": "step"})

        # add 'locidx' to the coordinates
        ds = ds.assign_coords(locidx=ds["locidx"])
        ds = ds.set_coords(["latitude", "longitude"])

        # add to the list of datasets
        datasets.append(ds)

    return datasets


def merged_zarrs(ds):
    ds_merged = xr.concat(ds, dim="ts")
    ds_merged = ds_merged.drop_vars("datetimes")

    var_names = ds_merged.data_vars
    d2 = xr.concat([ds_merged[v] for v in var_names], dim="variable")
    d2 = d2.assign_coords(variable=("variable", var_names))
    ds_merged = xr.Dataset(dict(value=d2))
    ds_merged = ds_merged.sortby("step")
    ds_merged = ds_merged.sortby("ts")

    ds_merged["step"] = (
        "step",
        np.array(ds_merged["step"].values, dtype="timedelta64[h]"),
    )

    return ds_merged


def main():
    args = _parse_args()

    output_path = f"{args.output}/excarta_{args.year}.zarr"

    # if args.output.exists() and not args.force:
    #     raise RuntimeError(f'Output file "{args.output}" already exist')

    datasets = extract_files(args)
    print("merging zarrs")
    ds_merged = merged_zarrs(datasets)
    print("zarrs merged")

    ds_merged.to_zarr(output_path)

    print("file saved at output_path")


if __name__ == "__main__":
    main()
