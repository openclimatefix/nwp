"""Convert ICON backlog kindly given to us for Hugging Face/our use

The data is all in one directory, so we need to split it by filename more

Only ICON EU, seems to have all the variables?

"""

import os
from glob import glob

import xarray as xr
import zarr
from huggingface_hub import HfApi
from ocf_blosc2 import Blosc2

from nwp.icon.consts import (
    EU_VAR2D_LIST,
    EU_VAR3D_LIST,
)


def process_model_files(folder, date, run="00"):
    filename_datetime = f"{date}{run}"
    var_base = "icon-eu_europe_regular-lat-lon"
    var_3d_list = EU_VAR3D_LIST
    var_2d_list = EU_VAR2D_LIST
    lons = None
    lats = None
    datasets = []
    for var_3d in var_3d_list:
        print(var_3d)
        paths = [
            list(
                glob(
                    os.path.join(
                        folder,
                        run,
                        f"{var_base}_pressure-level_{filename_datetime}_{str(s).zfill(3)}_*_{var_3d.upper()}.grib2",
                    )
                )
            )
            for s in range(73)
        ]
        try:
            ds = xr.concat(
                [
                    xr.open_mfdataset(
                        p,
                        engine="cfgrib",
                        backend_kwargs={"errors": "ignore"},
                        combine="nested",
                        concat_dim="isobaricInhPa",
                    ).sortby("isobaricInhPa")
                    for p in paths
                ],
                dim="step",
            ).sortby("step")
        except Exception as e:
            print(e)
            continue
        ds = ds.rename({v: var_3d for v in ds.data_vars})
        coords_to_remove = []
        for coord in ds.coords:
            if coord not in ds.dims and coord != "time" and coord != "valid_time":
                coords_to_remove.append(coord)
        print(coords_to_remove)
        if len(coords_to_remove) > 0:
            ds = ds.drop_vars(coords_to_remove)
        datasets.append(ds)
    ds_atmos = xr.merge(datasets)
    print(ds_atmos)
    total_dataset = []
    for var_2d in var_2d_list:
        print(var_2d)
        try:
            ds = xr.open_mfdataset(
                os.path.join(
                    folder,
                    run,
                    f"{var_base}_single-level_{filename_datetime}_*_{var_2d.upper()}.grib2",
                ),
                engine="cfgrib",
                combine="nested",
                concat_dim="step",
                backend_kwargs={"errors": "ignore"},
            ).sortby("step")
        except Exception as e:
            print(e)
            continue
        # Rename data variable to name in list, so no conflicts
        ds = ds.rename({v: var_2d for v in ds.data_vars})
        # Remove extra coordinates that are not dimensions or time
        coords_to_remove = []
        for coord in ds.coords:
            if coord not in ds.dims and coord != "time":
                coords_to_remove.append(coord)
        if len(coords_to_remove) > 0:
            ds = ds.drop_vars(coords_to_remove)
        total_dataset.append(ds)
    ds = xr.merge(total_dataset)
    print(ds)
    # Merge both
    ds = xr.merge([ds, ds_atmos])
    if lats is not None and lons is not None:
        ds = ds.assign_coords({"latitude": lats, "longitude": lons})
    print(ds)
    return ds


def upload_to_hf(dataset_xr, folder, model="eu", run="00", token=None):
    api = HfApi(token=token)
    zarr_path = os.path.join(folder, f"{run}.zarr.zip")
    if model == "global":
        chunking = {
            "step": 37,
            "values": 122500,
            "isobaricInhPa": -1,
        }
    else:
        chunking = {
            "step": 37,
            "latitude": 326,
            "longitude": 350,
            "isobaricInhPa": -1,
        }
    encoding = {var: {"compressor": Blosc2("zstd", clevel=9)} for var in dataset_xr.data_vars}
    encoding["time"] = {"units": "nanoseconds since 1970-01-01"}
    with zarr.ZipStore(
        zarr_path,
        mode="w",
    ) as store:
        dataset_xr.chunk(chunking).to_zarr(store, encoding=encoding, compute=True)
    done = False
    while not done:
        try:
            api.upload_file(
                path_or_fileobj=zarr_path,
                path_in_repo=f"data/{dataset_xr.time.dt.year.values}/"
                f"{dataset_xr.time.dt.month.values}/"
                f"{dataset_xr.time.dt.day.values}/"
                f"{dataset_xr.time.dt.year.values}{str(dataset_xr.time.dt.month.values).zfill(2)}{str(dataset_xr.time.dt.day.values).zfill(2)}"
                f"_{str(dataset_xr.time.dt.hour.values).zfill(2)}.zarr.zip",
                repo_id="openclimatefix/dwd-icon-global"
                if model == "global"
                else "openclimatefix/dwd-icon-eu",
                repo_type="dataset",
            )
            done = True
            try:
                os.remove(zarr_path)
            except:
                continue
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # Go through each date and process the data
    # Upload to HF
    folder = "/mnt/storage_c/ICON_DEXTER/"
    for year in [2021, 2022]:
        for month in range(1, 13):
            for day in range(1, 32):
                for run in ["00", "06", "12", "18"]:
                    date = f"{year}{str(month).zfill(2)}{str(day).zfill(2)}"
                    print(date)
                    ds = process_model_files(folder, date, run=run)
                    upload_to_hf(ds, folder, model="eu", run=run)
                    print(f"Uploaded {date}{run} to HF successfully")
