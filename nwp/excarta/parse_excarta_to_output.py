# import libs
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import os
import pathlib
from datetime import datetime
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path, help="Output zarr file")
    return parser.parse_args()

# sort the data into the correct format
def data_loader(folder_path):
    
    column_names = ['DateTimeUTC', 'LocationId', 'Latitude', 'Longitude', 'dni', 'dhi', 'ghi']
    files = os.listdir(folder_path)
    dfs = []
    
    for filename in files:
        if filename.endswith(".csv") and not filename.startswith("._"):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            print(filename)
            df = pd.read_csv(file_path, header=None, names=column_names, parse_dates=['DateTimeUTC'])
            
            # Remove the .csv
            datetime_str = filename[:-4] 
            # Convert the filename to a datetime object
            datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H")

            df['step'] = (df['DateTimeUTC'] - datetime_obj).dt.total_seconds() / 3600  # convert timedelta to hours
            df['init_time'] = datetime_obj
            dfs.append(df)
        
    return dfs
    
def load_data_from_all_years(parent_folder_path):
    # Initialize an empty list to store the dataframes
    all_dataframes = []

    # Loop over each year's folder and call the folder_data_load_sorted function
    for year in range(2018, 2019):
        folder_path = os.path.join(parent_folder_path, str(year))
        dataframes = data_loader(folder_path)
        all_dataframes.extend(dataframes)

    return all_dataframes


def pdtocdf(dfs):
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    ds = xr.Dataset.from_dataframe(merged_df)
    ds = ds.set_index(index=['init_time', 'step','Latitude','Longitude']).unstack('index')    
    # ds = ds.assign_coords(latitude=ds["Latitude"])
    # ds = ds.assign_coords(longitude=ds["Longitude"])
    
    # ds = ds.drop("Latitude")
    # ds = ds.drop("Longitude")
    
    ds = ds.drop_vars(["LocationId", "DateTimeUTC"])
    
    var_names = ds.data_vars
    d2 = xr.concat([ds[v] for v in var_names], dim="variable")

    # Set the coordinates to keep the names of the variables.
    d2 = d2.assign_coords(variable=("variable", var_names))

    # Turn the xr.DataArray into a xr.Dataset.
    ds = xr.Dataset(dict(value=d2))

    #When datesets are merged the steps can be out of order, so we sort them
    ds = ds.sortby('step')
    ds = ds.sortby('init_time')

    ds = ds.rename({"Latitude": "y", "Longitude": "x"})

    return ds



def main():

    args = _parse_args()

    if args.output.exists():
        raise RuntimeError(f'Output file "{args.output}" already exist')


    PATH = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/solar_data"
    dfs = load_data_from_all_years(PATH)
    ds = pdtocdf(dfs)
    ds.to_zarr(args.output)

main()
