# import libs
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import os
import pathlib as Path
from datetime import datetime
import zarr
import ocf_blosc2

def merge_zarr_files(zarr_path, merged_zarr_path):
    # Collect paths of Zarr files in the specified directory
    zarr_files = [os.path.join(zarr_path, file) for file in os.listdir(zarr_path) if file.endswith('.zarr')]

    print("1")
    # Open the first Zarr file to create the initial dataset
    merged_ds = xr.open_zarr(zarr_files[0])
    
    print("2")

    # Define the specific range of x and y coordinates
    # x_range = (-10, 2)  # Example x coordinate range
    # y_range = (49, 59)  # Example y coordinate range

    # Iterate over the remaining Zarr files and merge them into the initial dataset
    for file in zarr_files[1:]:
        ds = xr.open_zarr(file)
        print(file)

        # ds_filt = ds.sel(x=slice(*x_range), y=slice(*y_range))
        merged_ds = merged_ds.combine_first(ds_filt)
        
    print("3")

    # Rechunk the merged dataset
    merged_ds = merged_ds.chunk(chunks={"init_time": 10, "x": 100, "y": 100})
    
    print("4")
    


    
    print(merged_ds)

    # Save the merged dataset as a new Zarr file
    merged_ds.to_zarr(merged_zarr_path)
    
    print("5")
    
    


# Specify the path where the independent Zarr files are located
zarr_path = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/zarr_format/r3"

# Specify the path for the merged Zarr file
merged_zarr_path = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/merged_excarta/merged_r3_UK_full_t1.zarr"

# Merge the Zarr files
merge_zarr_files(zarr_path, merged_zarr_path)

