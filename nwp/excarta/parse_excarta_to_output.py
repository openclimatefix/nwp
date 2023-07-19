import argparse
import datetime
import os
import pathlib
from datetime import datetime

import pandas as pd
import xarray as xr


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Output zarr file")
    return parser.parse_args()


def data_loader(folder_path):
    """
    Loads and transforms data from CSV files in the given folder_path.
    """
    column_names = ["DateTimeUTC", "LocationId", "Latitude", "Longitude", "dni", "dhi", "ghi"]
    files = os.listdir(folder_path)
    dfs = []

    for filename in files:
        if filename.endswith(".csv") and not filename.startswith("._"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(
                file_path, header=None, names=column_names, parse_dates=["DateTimeUTC"]
            )

            datetime_str = filename[:-4]
            datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H")

            df["step"] = (
                df["DateTimeUTC"] - datetime_obj
            ).dt.total_seconds() / 3600  # convert timedelta to hours
            df["init_time"] = datetime_obj
            dfs.append(df)

    return dfs


def load_data_from_all_years(parent_folder_path):
    """
    Loads data from all the year folders under the parent path.
    """
    all_dataframes = []

    # Actual date range is 2018 to 2022 (for in range use (2018,2023))
    for year in range(2018, 2019):
        folder_path = os.path.join(parent_folder_path, str(year))
        dataframes = data_loader(folder_path)
        all_dataframes.extend(dataframes)

    return all_dataframes


def pdtocdf(dfs):
    """
    Converts pandas dataframe to an xarray dataset.
    """
    merged_df = pd.concat(dfs, ignore_index=True)

    ds = xr.Dataset.from_dataframe(merged_df)
    ds = ds.set_index(index=["init_time", "step", "Latitude", "Longitude"]).unstack("index")
    ds = ds.drop_vars(["LocationId", "DateTimeUTC"])

    var_names = ds.data_vars
    d2 = xr.concat([ds[v] for v in var_names], dim="variable")
    d2 = d2.assign_coords(variable=("variable", var_names))
    ds = xr.Dataset(dict(value=d2))
    ds = ds.sortby("step")
    ds = ds.sortby("init_time")
    ds = ds.rename({"Latitude": "y", "Longitude": "x"})

    return ds


def main():
    """
    Main function to control the flow of the script.
    """
    args = _parse_args()

    if args.output.exists():
        raise RuntimeError(f'Output file "{args.output}" already exist')

    PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/solar_data"
    dfs = load_data_from_all_years(PATH)
    ds = pdtocdf(dfs)
    ds.to_zarr(args.output)


# Check if script is being run directly
if __name__ == "__main__":
    main()
