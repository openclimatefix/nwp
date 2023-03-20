import os
import xarray as xr
from glob import glob
from ocf_blosc2 import Blosc2
from huggingface_hub import HfApi
from ocf_blosc2 import Blosc2
import zarr
import shutil
import argparse
from icon.consts import *
from icon.utils import get_dset
import click

api = HfApi()


def download_model_files(runs=None, parent_folder=None, model="global"):
    """

    """
    if runs is None:
        runs = ["00", "06", "12", "18", ]
    if model == "global":
        var_3d_list = GLOBAL_VAR3D_LIST
        var_2d_list = GLOBAL_VAR2D_LIST
        invariant = GLOBAL_INVARIENT_LIST
        pressure_levels = GLOBAL_PRESSURE_LEVELS
    else:
        var_3d_list = EU_VAR3D_LIST
        var_2d_list = EU_VAR2D_LIST
        invariant = None
        pressure_levels = EU_PRESSURE_LEVELS
    for run in runs:
        run_folder = f"{parent_folder}/{run}/"
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)

        f_steps = list(range(0, 73))
        vars_3d = [v + "@" + str(p) for v in var_3d_list for p in pressure_levels]
        vars_2d = var_2d_list
        not_done = True
        while not_done:
            try:
                get_dset(vars_2d=vars_2d, vars_3d=vars_3d, invarient=invariant, folder=run_folder, run=run,
                         f_times=f_steps)
                not_done = False
            except:
                continue


def process_model_files(folder, var_3d_list=None, var_2d_list=None, invariant_list=None, model="global"):
    if model == "global":
        hf_path = "openclimatefix/dwd-icon-global"
        var_base = "icon_global_icosahedral"
        var_3d_list = GLOBAL_VAR3D_LIST
        var_2d_list = GLOBAL_VAR2D_LIST
        invariant_list = GLOBAL_INVARIENT_LIST
    else:
        hf_path = "openclimatefix/dwd-icon-eu"
        var_base = "icon-eu_europe_regular-lat-lon"
        var_3d_list = EU_VAR3D_LIST
        var_2d_list = EU_VAR2D_LIST
        invariant_list = None
    if invariant_list is not None:
        lon_ds = xr.open_dataset(
            list(glob(f"{folder}/icon_global_icosahedral_time-invariant_*_CLON.grib2"))[0],
            engine="cfgrib")
        lat_ds = xr.open_dataset(
            list(glob(f"{folder}/icon_global_icosahedral_time-invariant_*_CLAT.grib2"))[0],
            engine="cfgrib")
        lons = lon_ds.tlon.values
        lats = lat_ds.tlat.values
    else:
        lons = None
        lats = None
    datasets = []
    for var_3d in var_3d_list:
        print(var_3d)
        paths = [list(glob(
            f"{folder}/{var_base}_pressure-level_*_{str(s).zfill(3)}_*_{var_3d.upper()}.grib2"))
            for s in range(73)]
        try:
            ds = xr.concat(
                [xr.open_mfdataset(p, engine="cfgrib", combine="nested", concat_dim="isobaricInhPa").sortby(
                    "isobaricInhPa") for p in paths], dim="step").sortby("step")
        except Exception as e:
            print(e)
            continue
        ds = ds.rename({v: var_3d for v in ds.data_vars})
        coords_to_remove = []
        for coord in ds.coords:
            if coord not in ds.dims and coord != "time":
                coords_to_remove.append(coord)
        if len(coords_to_remove) > 0:
            ds = ds.drop_vars(coords_to_remove)
        datasets.append(ds)
    ds_atmos = xr.merge(datasets)
    print(ds_atmos)
    files = api.list_repo_files(hf_path, repo_type="dataset")
    if f"data/{ds_atmos.time.dt.year.values}/{ds_atmos.time.dt.month.values}/{ds_atmos.time.dt.day.values}/{ds_atmos.time.dt.year.values}{str(ds_atmos.time.dt.month.values).zfill(2)}{str(ds_atmos.time.dt.day.values).zfill(2)}_{str(ds_atmos.time.dt.hour.values).zfill(2)}.zarr.zip" in files:
        return None

    total_dataset = []
    for var_2d in var_2d_list:
        print(var_2d)
        try:
            ds = xr.open_mfdataset(
                f"{folder}/{var_base}_single-level_*_*_{var_2d.upper()}.grib2",
                engine="cfgrib", combine="nested", concat_dim="step").sortby("step").drop_vars("valid_time")
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


def upload_to_hf(dataset_xr, folder, model="global", run="00"):
    zarr_path = f"{folder}/{run}.zarr.zip"
    if model == "global":
        chunking = {"step": 37, "values": 122500, "isobaricInhPa": -1, }
    else:
        chunking = {"step": 37, "latitude": 326, "longitude": 350, "isobaricInhPa": -1, }
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
                path_in_repo=f"data/{dataset_xr.time.dt.year.values}/{dataset_xr.time.dt.month.values}/{dataset_xr.time.dt.day.values}/{dataset_xr.time.dt.year.values}{str(dataset_xr.time.dt.month.values).zfill(2)}{str(dataset_xr.time.dt.day.values).zfill(2)}_{str(dataset_xr.time.dt.hour.values).zfill(2)}.zarr.zip",
                repo_id="openclimatefix/dwd-icon-global",
                repo_type="dataset",
            )
            done = True
            os.remove(zarr_path)
            shutil.rmtree(folder)
        except Exception as e:
            print(e)


@click.command()
@click.option(
    "--model",
    default=(
        "global"
    ),
    help=(
        "Model type, either 'global' or 'eu' "
    ),
)
@click.option(
    "--folder",
    default=("/mnt/storage_ssd_4tb/DWD/"),
    help="Folder to put the raw and zarr in",
)
@click.option(
    "--run",
    default=None,
    help=(
        "Run number to use, one of '00', '06', '12', '18', or leave off for all."
    ),
)
def main(
     model: str,
    folder: str,
    run: str,
):
    """The entry point into the script."""
    if run is not None:
        run = [run]
    elif run is None:
        run = ["00", "06", "12", "18", ]
    print(f"------------------- Downloading Model Files for: {run=}")
    download_model_files(runs=run, parent_folder=folder, model=model)
    for r in run:
        run_folder = f"{folder}/{r}/"
        print(f"--------------------- Processing Model Files For {r}")
        ds = process_model_files(folder=run_folder, model=model)
        if ds is not None:
            print(f"--------------------- Uploading to HuggingFace Run: {r}")
            upload_to_hf(ds, folder=run_folder, model=model, run=r)


if __name__ == "__main__":
    main()
