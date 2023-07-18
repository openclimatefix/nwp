#Low memory script
import os
from datetime import datetime
import pandas as pd
import xarray as xr
import argparse
import pathlib

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Output zarr file")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    return parser.parse_args()



def data_loader(folder_path, month_to_process):
    """
    Loads and transforms data from CSV files in the given folder_path and directly convert each DataFrame into an xarray Dataset.
    Only process files for the month 'YYYYMM' given by month_to_process
    """
    month_to_process = datetime.strptime(month_to_process, "%Y%m")
    column_names = ['DateTimeUTC', 'LocationId', 'Latitude', 'Longitude', 'dni', 'dhi', 'ghi']
    files = os.listdir(folder_path)
    datasets = []

    for filename in files:
        if filename.endswith(".csv") and not filename.startswith("._"):
            file_datetime = datetime.strptime(filename[:-4], "%Y%m%d%H")

            if (file_datetime.year == month_to_process.year) and (file_datetime.month == month_to_process.month):

                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, header=None, names=column_names, parse_dates=['DateTimeUTC'])
    
                df['step'] = (df['DateTimeUTC'] - file_datetime).dt.total_seconds() / 3600  # convert timedelta to hours
                df['init_time'] = file_datetime

                # Convert the dataframe to an xarray Dataset and append to the list
                ds = xr.Dataset.from_dataframe(df)
                ds = ds.drop_vars(["LocationId", "DateTimeUTC"])
                datasets.append(ds)

    return datasets


def load_data_from_all_years(parent_folder_path, month_to_process):
    all_datasets = []

    # Get 'year' part from month_to_process 'YYYYMM' in string format
    year_to_process = int(month_to_process[:4])

    folder_path = os.path.join(parent_folder_path, str(year_to_process))
    datasets = data_loader(folder_path, month_to_process)
    all_datasets.extend(datasets)

    return all_datasets


def pdtocdf(datasets):
    """
    Processes the xarray Datasets and merges them.
    """
    
    datasets = [ds.set_index(index=['init_time', 'step', 'Latitude', 'Longitude']) for ds in datasets]

    ds = xr.concat(datasets, dim='index')

    var_names = ds.data_vars
    d2 = xr.concat([ds[v] for v in var_names], dim="variable")
    d2 = d2.assign_coords(variable=("variable", var_names))
    ds = xr.Dataset(dict(value=d2))
    ds = ds.sortby('step')
    ds = ds.sortby('init_time')
    ds = ds.rename({"Latitude": "y", "Longitude": "x"})

    return ds


def main():
    args = _parse_args()

    if args.output.exists():
        raise RuntimeError(f'Output file "{args.output}" already exist')

    PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/solar_data"
    month_to_process = f"{args.year}{args.month:02d}"  # combine year and month arguments into the required format
    datasets = load_data_from_all_years(PATH, month_to_process)
    ds = pdtocdf(datasets)

    print(ds)
    ds = ds.unstack('index')

    file_ending = ".zarr"

    # Create output directory name including the year and month to process
    output_name = f"{args.output}{args.year}{args.month:02d}{file_ending}"
    ds.to_zarr(output_name)


# Check if script is being run directly
if __name__ == "__main__":
    main()