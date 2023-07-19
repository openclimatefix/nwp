# Low memory script
import os
from datetime import datetime
import pandas as pd
import xarray as xr
import argparse
import pathlib


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Output zarr file")
    return parser.parse_args()


def data_loader(folder_path):
    """
    Loads and transforms data from CSV files in the given folder_path and directly convert each DataFrame into an xarray Dataset.
    """
    column_names = [
        "DateTimeUTC",
        "LocationId",
        "Latitude",
        "Longitude",
        "dni",
        "dhi",
        "ghi",
    ]
    files = os.listdir(folder_path)
    datasets = []

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

            # Convert the dataframe to an xarray Dataset and append to the list
            ds = xr.Dataset.from_dataframe(df)
            ds = ds.drop_vars(["LocationId", "DateTimeUTC"])
            datasets.append(ds)

    return datasets


def load_data_from_all_years(parent_folder_path):
    all_datasets = []

    for year in range(2017, 2019):
        folder_path = os.path.join(parent_folder_path, str(year))
        datasets = data_loader(folder_path)
        all_datasets.extend(datasets)

    return all_datasets


def pdtocdf(datasets):
    """
    Processes the xarray Datasets and merges them.
    """
    print(datasets)
    #     ds = xr.merge(datasets)

    datasets = [
        ds.set_index(index=["init_time", "step", "Latitude", "Longitude"])
        for ds in datasets
    ]

    ds = xr.concat(datasets, dim="index")

    # Going to unstack and then combine in a different script
    # Get rid of the index dimension and just keep the desired ones
    # ds = ds.unstack('index')

    var_names = ds.data_vars
    d2 = xr.concat([ds[v] for v in var_names], dim="variable")
    d2 = d2.assign_coords(variable=("variable", var_names))
    ds = xr.Dataset(dict(value=d2))
    ds = ds.sortby("step")
    ds = ds.sortby("init_time")
    ds = ds.rename({"Latitude": "y", "Longitude": "x"})

    return ds


def main():
    args = _parse_args()

    if args.output.exists():
        raise RuntimeError(f'Output file "{args.output}" already exist')

    PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/solar_data"
    datasets = load_data_from_all_years(PATH)
    ds = pdtocdf(datasets)

    print(ds)

    ds = ds.unstack("index")

    ds.to_zarr(args.output)


# Check if script is being run directly
if __name__ == "__main__":
    main()
