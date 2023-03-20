import bz2
import datetime
import os
import shutil
import time
from datetime import datetime
from glob import glob
from multiprocessing import Pool, cpu_count

import requests
import xarray as xr
import zarr
from huggingface_hub import HfApi
from ocf_blosc2 import Blosc2

api = HfApi()
files = api.list_repo_files("openclimatefix/dwd-icon-global", repo_type="dataset")

"""
# CDO grid description file for global regular grid of ICON.
gridtype  = lonlat
xsize     = 2399
ysize     = 1199
xfirst    = -180
xinc      = 0.15
yfirst    = -90
yinc      = 0.15
"""


var_2d_list = [
    "alb_rad",
    "alhfl_s",
    "ashfl_s",
    "asob_s",
    "asob_t",
    "aswdifd_s",
    "aswdifu_s",
    "aswdir_s",
    "athb_s",
    "athb_t",
    "aumfl_s",
    "avmfl_s",
    "cape_con",
    "cape_ml",
    "clch",
    "clcl",
    "clcm",
    "clct",
    "clct_mod",
    "cldepth",
    "c_t_lk",
    "freshsnw",
    "fr_ice",
    "h_snow",
    "h_ice",
    "h_ml_lk",
    "hbas_con",
    "htop_con",
    "htop_dc",
    "hzerocl",
    "pmsl",
    "ps",
    "qv_2m",
    "qv_s",
    "rain_con",
    "rain_gsp",
    "relhum_2m",
    "rho_snow",
    "runoff_g",
    "runoff_s",
    "snow_con",
    "snow_gsp",
    "snowlmt",
    "synmsg_bt_cl_ir10.8",
    "t_2m",
    "t_g",
    "t_snow",
    "t_ice",
    "t_s",
    "tch",
    "tcm",
    "td_2m",
    "tmax_2m",
    "tmin_2m",
    "tot_prec",
    "tqc",
    "tqi",
    "tqr",
    "tqs",
    "tqv",
    "u_10m",
    "v_10m",
    "vmax_10m",
    "w_snow",
    "w_so",
    "ww",
    "z0",
]

var_3d_list = ["clc", "fi", "p", "qv", "relhum", "t", "tke", "u", "v", "w"]

invarient_list = [
    "clat",
    "clon",
]

pressure_levels = [
    1000,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
]

while True:
    now = datetime.now()
    if now.hour in [3, 9, 15, 21]:
        for run in ["00", "06", "12", "18"]:
            if not os.path.exists(
                f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/"
            ):
                os.mkdir(
                    f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/"
                )

            def get_run():
                now = datetime.now()

                return now.strftime("%Y%m%d") + run, run

            def find_file_name(
                vars_2d=None,
                vars_3d=None,
                f_times=0,
                base_url="https://opendata.dwd.de/weather/nwp",
                model_url="icon/grib",
            ):
                """Find file names to be downloaded given input variables and
                a forecast lead time f_time (in hours).
                - vars_2d, a list of 2d variables to download, e.g. ['t_2m']
                - vars_3d, a list of 3d variables to download with pressure
                  level, e.g. ['t@850','fi@500']
                - f_times, forecast steps, e.g. 0 or list(np.arange(1, 79))
                Note that this function WILL NOT check if the files exist on
                the server to avoid wasting time. When they're passed
                to the download_extract_files function if the file does not
                exist it will simply not be downloaded.
                """
                date_string, run_string = get_run()
                if type(f_times) is not list:
                    f_times = [f_times]
                if (vars_2d is None) and (vars_3d is None):
                    raise ValueError("You need to specify at least one 2D or one 3D variable")

                if vars_2d is not None:
                    if type(vars_2d) is not list:
                        vars_2d = [vars_2d]
                if vars_3d is not None:
                    if type(vars_3d) is not list:
                        vars_3d = [vars_3d]

                urls = []
                for f_time in f_times:
                    if vars_2d is not None:
                        for var in vars_2d:
                            if var not in var_2d_list:
                                raise ValueError("accepted 2d variables are %s" % var_2d_list)
                            var_url = "icon_global_icosahedral_single-level"
                            urls.append(
                                f"{base_url}/{model_url}/{run_string}/{var}/{var_url}_{date_string}_%03d_{var.upper()}.grib2.bz2"
                                % (f_time,)
                            )
                    if vars_3d is not None:
                        for var in vars_3d:
                            var_t, plev = var.split("@")
                            if var_t not in var_3d_list:
                                raise ValueError("accepted 3d variables are %s" % var_3d_list)
                            var_url = "icon_global_icosahedral_pressure-level"
                            urls.append(
                                f"{base_url}/{model_url}/{run_string}/{var_t}/{var_url}_{date_string}_%03d_{plev}_{var_t.upper()}.grib2.bz2"
                                % (f_time,)
                            )
                    for var in invarient_list:
                        var_url = "icon_global_icosahedral_time-invariant"
                        urls.append(
                            f"{base_url}/{model_url}/{run_string}/{var}/{var_url}_{date_string}_{var.upper()}.grib2.bz2"
                        )
                return urls

            def download_extract_files(urls):
                """Given a list of urls download and bunzip2 them.
                Return a list of the path of the extracted files
                """

                if type(urls) is list:
                    urls_list = urls
                else:
                    urls_list = [urls]

                # We only parallelize if we have a number of files
                # larger than the cpu count
                if len(urls_list) > cpu_count():
                    pool = Pool(cpu_count())
                    results = pool.map(download_extract_url, urls_list)
                    pool.close()
                    pool.join()
                else:
                    results = []
                    for url in urls_list:
                        results.append(download_extract_url(url))

                return results

            def download_extract_url(
                url,
                folder=f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/",
            ):
                filename = folder + os.path.basename(url).replace(".bz2", "")

                if os.path.exists(filename):
                    extracted_files = filename
                else:
                    r = requests.get(url, stream=True)
                    if r.status_code == requests.codes.ok:
                        with r.raw as source, open(filename, "wb") as dest:
                            dest.write(bz2.decompress(source.read()))
                        extracted_files = filename
                    else:
                        return None

                return extracted_files

            def get_dset(vars_2d=[], vars_3d=[], f_times=0):
                if vars_2d or vars_3d:
                    date_string, _ = get_run()
                    urls = find_file_name(vars_2d=vars_2d, vars_3d=vars_3d, f_times=f_times)
                    fils = download_extract_files(urls)

                return fils

            f_steps = list(range(0, 73))
            vars_3d_download = var_3d_list
            pressure_levels_download = pressure_levels
            vars_3d = [v + "@" + str(p) for v in vars_3d_download for p in pressure_levels_download]
            vars_2d = var_2d_list

            not_done = True
            while not_done:
                try:
                    get_dset(vars_2d=vars_2d, vars_3d=vars_3d, f_times=f_steps)
                    not_done = False
                except Exception as e:
                    print(e)
                    continue

        for run in ["00", "06", "12", "18"]:
            lon_ds = xr.open_mfdataset(
                f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/icon_global_icosahedral_time-invariant_*_CLON.grib2",
                engine="cfgrib",
            )
            lat_ds = xr.open_mfdataset(
                f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/icon_global_icosahedral_time-invariant_*_CLAT.grib2",
                engine="cfgrib",
            )
            lons = lon_ds.tlon.values
            lats = lat_ds.tlat.values
            datasets = []
            for var_3d in var_3d_list:
                paths = [
                    list(
                        glob(
                            f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/icon_global_icosahedral_pressure-level_*_{str(s).zfill(3)}_*_{var_3d.upper()}.grib2"
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
                    if coord not in ds.dims and coord != "time":
                        coords_to_remove.append(coord)
                if len(coords_to_remove) > 0:
                    ds = ds.drop_vars(coords_to_remove)
                datasets.append(ds)
                print(ds)
            ds_atmos = xr.merge(datasets)
            files = api.list_repo_files("openclimatefix/dwd-icon-global", repo_type="dataset")
            if (
                f"data/{ds_atmos.time.dt.year.values}/{ds_atmos.time.dt.month.values}/{ds_atmos.time.dt.day.values}/{ds_atmos.time.dt.year.values}{ds_atmos.time.dt.month.values}{ds_atmos.time.dt.day.values}_{ds_atmos.time.dt.hour.values}.zarr.zip"
                in files
            ):
                continue

            total_dataset = []
            for var_2d in var_2d_list:
                datasets = []
                print(var_2d)
                try:
                    ds = (
                        xr.open_mfdataset(
                            f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/icon_global_icosahedral_single-level_*_*_{var_2d.upper()}.grib2",
                            engine="cfgrib",
                            backend_kwargs={"errors": "ignore"},
                            combine="nested",
                            concat_dim="step",
                        )
                        .sortby("step")
                        .drop_vars("valid_time")
                    )
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
                print(ds)
                total_dataset.append(ds)
            ds = xr.merge(total_dataset)
            print(ds)
            # Merge both
            ds = xr.merge([ds, ds_atmos])
            ds = ds.assign_coords({"latitude": lats, "longitude": lons})
            print(ds)
            encoding = {var: {"compressor": Blosc2("zstd", clevel=9)} for var in ds.data_vars}
            encoding["time"] = {"units": "nanoseconds since 1970-01-01"}
            with zarr.ZipStore(
                f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}.zarr.zip",
                mode="w",
            ) as store:
                ds.chunk(
                    {
                        "step": 37,
                        "values": 122500,
                        "isobaricInhPa": -1,
                    }
                ).to_zarr(store, encoding=encoding, compute=True)
            done = False
            while not done:
                try:
                    api.upload_file(
                        path_or_fileobj=f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}.zarr.zip",
                        path_in_repo=f"data/{ds.time.dt.year.values}/{ds.time.dt.month.values}/{ds.time.dt.day.values}/{ds.time.dt.year.values}{ds.time.dt.month.values}{ds.time.dt.day.values}_{ds.time.dt.hour.values}.zarr.zip",
                        repo_id="openclimatefix/dwd-icon-global",
                        repo_type="dataset",
                    )
                    done = True
                    shutil.rmtree(
                        f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}/"
                    )
                    os.remove(
                        f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_Global/{run}.zarr.zip"
                    )
                except Exception as e:
                    print(e)
            del ds
            del ds_atmos
        time.sleep(60 * 60 * 1)  # 1 hour wait once done
