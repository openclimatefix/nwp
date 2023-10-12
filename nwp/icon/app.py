import os
import shutil
from glob import glob

import click
import xarray as xr
import zarr
from huggingface_hub import HfApi
from ocf_blosc2 import Blosc2

from nwp.icon.consts import (
    EU_PRESSURE_LEVELS,
    EU_VAR2D_LIST,
    EU_VAR3D_LIST,
    GLOBAL_INVARIENT_LIST,
    GLOBAL_PRESSURE_LEVELS,
    GLOBAL_VAR2D_LIST,
    GLOBAL_VAR3D_LIST,
)
from nwp.icon.utils import get_dset, get_run


def download_model_files(runs=None, parent_folder=None, model="global", delay=0):
    if runs is None:
        runs = [
            "00",
            "06",
            "12",
            "18",
        ]
    if model == "global":
        var_3d_list = GLOBAL_VAR3D_LIST
        var_2d_list = GLOBAL_VAR2D_LIST
        invariant = GLOBAL_INVARIENT_LIST
        pressure_levels = GLOBAL_PRESSURE_LEVELS
        f_steps = list(range(0, 79)) + list(range(81, 99, 3))  # 4 days
    else:
        var_3d_list = EU_VAR3D_LIST
        var_2d_list = EU_VAR2D_LIST
        invariant = None
        pressure_levels = EU_PRESSURE_LEVELS
        f_steps = list(range(0, 79)) + list(range(81, 123, 3))  # 5 days
    for run in runs:
        run_folder = os.path.join(parent_folder, run)
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)

        vars_3d = [v + "@" + str(p) for v in var_3d_list for p in pressure_levels]
        vars_2d = var_2d_list
        not_done = True
        tries = 0
        while not_done and tries < 10:
            try:
                tries += 1
                print(f"Tries: {tries}")
                get_dset(
                    vars_2d=vars_2d,
                    vars_3d=vars_3d,
                    invarient=invariant,
                    folder=run_folder,
                    run=run,
                    f_times=f_steps,
                    model=model,
                    delay=delay,
                )
                not_done = False
            except Exception as e:
                print(e)
                continue


def process_model_files(
    folder,
    var_3d_list=None,
    var_2d_list=None,
    invariant_list=None,
    model="global",
    run="00",
    delay=0,
):
    date_string, _ = get_run(run, delay=delay)
    if model == "global":
        var_base = "icon_global_icosahedral"
        var_3d_list = GLOBAL_VAR3D_LIST
        var_2d_list = GLOBAL_VAR2D_LIST
        lon_ds = xr.open_dataset(
            list(
                glob(
                    os.path.join(folder, run, f"{var_base}_time-invariant_{date_string}_CLON.grib2")
                )
            )[0],
            engine="cfgrib",
            backend_kwargs={"errors": "ignore"},
        )
        lat_ds = xr.open_dataset(
            list(
                glob(
                    os.path.join(folder, run, f"{var_base}_time-invariant_{date_string}_CLAT.grib2")
                )
            )[0],
            engine="cfgrib",
            backend_kwargs={"errors": "ignore"},
        )
        lons = lon_ds.tlon.values
        lats = lat_ds.tlat.values
        f_steps = list(range(0, 79)) + list(range(81, 99, 3))  # 4 days
    else:
        var_base = "icon-eu_europe_regular-lat-lon"
        var_3d_list = EU_VAR3D_LIST
        var_2d_list = EU_VAR2D_LIST
        lons = None
        lats = None
        f_steps = list(range(0, 79)) + list(range(81, 123, 3))
    datasets = []
    for var_3d in var_3d_list:
        print(var_3d)
        paths = [
            list(
                glob(
                    os.path.join(
                        folder,
                        run,
                        f"{var_base}_pressure-level_{date_string}_{str(s).zfill(3)}_*_{var_3d.upper()}.grib2",
                    )
                )
            )
            for s in f_steps
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
                    folder, run, f"{var_base}_single-level_{date_string}_*_{var_2d.upper()}.grib2"
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


def upload_to_hf(dataset_xr, folder, model="global", run="00", token=None):
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


def remove_files(folder: str) -> None:
    """
    Remove files in folder

    Args:
        folder: Folder to delete

    Returns:
        None
    """
    try:
        shutil.rmtree(folder)
    except:
        pass


@click.command()
@click.option(
    "--model",
    default=("global"),
    help=("Model type, either 'global' or 'eu' "),
)
@click.option(
    "--folder",
    default=("/run/media/jacob/data/DWD_Global/"),
    help="Folder to put the raw and zarr in",
)
@click.option(
    "--run",
    default=None,
    help=("Run number to use, one of '00', '06', '12', '18', or leave off for all."),
)
@click.option(
    "--delete",
    is_flag=True,
    default=False,
    help=("Whether to delete the run foldder files or not"),
)
@click.option(
    "--token",
    default=None,
    help=("HuggingFace Token"),
)
@click.option(
    "--timedelta",
    default=0,
    help=("Timedelta in days to get older data (either 1 day delay for yesterday, or 0 for today)"),
)
def main(model: str, folder: str, run: str, delete: bool, token: str, timedelta: int):
    """The entry point into the script."""
    assert model in ["global", "eu"]
    if run is not None:
        run = [run]
    elif run is None:
        run = [
            "00",
            "06",
            "12",
            "18",
        ]
    if delete:
        print(f"----------------- Removing Model Files for : {model=} {run=}")
        for r in run:
            remove_files(os.path.join(folder, r))
    print(f"------------------- Downloading Model Files for: {model=} {run=}")
    if not os.path.exists(folder):
        os.mkdir(folder)
    download_model_files(runs=run, parent_folder=folder, model=model, delay=timedelta)
    for r in run:
        print(f"--------------------- Processing Model Files For {model=} {r}")
        ds = process_model_files(folder=folder, model=model, run=r, delay=timedelta)
        if ds is not None:
            print(f"--------------------- Uploading to HuggingFace Run: {model=} {r}")
            upload_to_hf(ds, folder=folder, model=model, run=r, token=token)
    if delete:
        print(f"---------------------- Removing Model Files for : {model=} {run=}")
        for r in run:
            remove_files(os.path.join(folder, r))


if __name__ == "__main__":
    main()
