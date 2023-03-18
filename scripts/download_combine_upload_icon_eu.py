import os
import xarray as xr
from datetime import datetime
import requests
import bz2
from multiprocessing import Pool, cpu_count
from glob import glob
from ocf_blosc2 import Blosc2
from huggingface_hub import HfApi
from ocf_blosc2 import Blosc2
import zarr
import shutil

api = HfApi()
files = api.list_repo_files("openclimatefix/dwd-icon-eu", repo_type="dataset")


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
    "h_snow",
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
    "tch",
    "tcm",
    "td_2m",
    "tmax_2m",
    "tmin_2m",
    "tot_prec",
    "tqc",
    "tqi",
    "u_10m",
    "v_10m",
    "vmax_10m",
    "w_snow",
    "w_so",
    "ww",
    "z0",
]

var_3d_list = ["clc", "fi", "omega", "p", "qv", "relhum", "t", "tke", "u", "v", "w"]

pressure_levels = [
    1000,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
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


for run in ["00", "06", "12", "18",]:
    if not os.path.exists(f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/"):
        os.mkdir(f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/")
    def get_run():
        now = datetime.now()

        return now.strftime("%Y%m%d") + run, run


    def find_file_name(
        vars_2d=None,
        vars_3d=None,
        f_times=0,
        base_url="https://opendata.dwd.de/weather/nwp",
        model_url="icon-eu/grib",
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
                    var_url = "icon-eu_europe_regular-lat-lon_single-level"
                    urls.append(
                        "%s/%s/%s/%s/%s_%s_%03d_%s.grib2.bz2"
                        % (
                            base_url,
                            model_url,
                            run_string,
                            var,
                            var_url,
                            date_string,
                            f_time,
                            var.upper(),
                        )
                    )
            if vars_3d is not None:
                for var in vars_3d:
                    var_t, plev = var.split("@")
                    if var_t not in var_3d_list:
                        raise ValueError("accepted 3d variables are %s" % var_3d_list)
                    var_url = "icon-eu_europe_regular-lat-lon_pressure-level"
                    urls.append(
                        "%s/%s/%s/%s/%s_%s_%03d_%s_%s.grib2.bz2"
                        % (
                            base_url,
                            model_url,
                            run_string,
                            var_t,
                            var_url,
                            date_string,
                            f_time,
                            plev,
                            var_t.upper(),
                        )
                    )

        return urls


    def download_extract_files(urls):
        """Given a list of urls download and bunzip2 them.
        Return a list of the path of the extracted files"""

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


    def download_extract_url(url, folder=f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/"):
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

    # get_dset(vars_2d=vars_2d,
    #          vars_3d=vars_3d,
    #          f_times=f_steps)
    not_done = True
    while not_done:
        try:
            get_dset(vars_2d=vars_2d, vars_3d=vars_3d, f_times=f_steps)
            not_done = False
        except:
            continue

# After downloading everything, then process and upload

for run in ["00", "06", "12", "18", ]:
    datasets = []
    for var_3d in var_3d_list:
        paths = [list(glob(
            f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/icon-eu_europe_regular-lat-lon_pressure-level_*_{str(s).zfill(3)}_*_{var_3d.upper()}.grib2"))
                 for s in range(73)]
        try:
            ds = xr.concat([xr.open_mfdataset(p, engine="cfgrib", combine="nested", concat_dim="isobaricInhPa").sortby(
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
        print(ds)
    ds_atmos = xr.merge(datasets)

    total_dataset = []
    for var_2d in var_2d_list:
        datasets = []
        print(var_2d)
        try:
            ds = xr.open_mfdataset(
                f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/icon-eu_europe_regular-lat-lon_single-level_*_*_{var_2d.upper()}.grib2",
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
        print(ds)
        total_dataset.append(ds)
    ds = xr.merge(total_dataset)
    print(ds)
    # Merge both
    ds = xr.merge([ds, ds_atmos])
    print(ds)
    encoding = {var: {"compressor": Blosc2("zstd", clevel=9)} for var in ds.data_vars}
    encoding["time"] = {"units": "nanoseconds since 1970-01-01"}
    with zarr.ZipStore(
            f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}.zarr.zip",
            mode="w",
    ) as store:
        ds.chunk({"step": 37, "latitude": 326, "longitude": 350, "isobaricInhPa": -1, }).to_zarr(store,
                                                                                                 encoding=encoding,
                                                                                            compute=True)
    done = False
    while not done:
        try:
            api.upload_file(
                path_or_fileobj=f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}.zarr.zip",
                path_in_repo=f"data/{ds.time.dt.year.values}/{ds.time.dt.month.values}/{ds.time.dt.day.values}/{ds.time.dt.year.values}{ds.time.dt.month.values}{ds.time.dt.day.values}_{ds.time.dt.hour.values}.zarr.zip",
                repo_id="openclimatefix/dwd-icon-eu",
                repo_type="dataset",
            )
            done = True
            shutil.rmtree(f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}/")
            os.remove(f"/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/DWD/ICON_EU/{run}.zarr.zip")
        except Exception as e:
            print(e)

