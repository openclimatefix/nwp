"""Utilities for downloading the DWD ICON models"""
import bz2
import os
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool, cpu_count

import requests


def get_run(run: str) -> tuple[str | str, str]:
    """
    Get run name

    Args:
        run: Run number

    Returns:
        Run date and run number
    """
    now = datetime.now()
    return now.strftime("%Y%m%d") + run, run


def find_file_name(
    vars_2d=None,
    vars_3d=None,
    invarient=None,
    f_times=0,
    base_url="https://opendata.dwd.de/weather/nwp",
    model_url="icon/grib",
    var_url_base="icon_global_icosahedral",
    run="00",
) -> list:
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
    date_string, run_string = get_run(run)
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
                var_url = f"{var_url_base}_single-level"
                urls.append(
                    f"{base_url}/{model_url}/{run_string}/{var}/{var_url}_{date_string}_{str(f_time).zfill(3)}_{var.upper()}.grib2.bz2"
                )
        if vars_3d is not None:
            for var in vars_3d:
                var_t, plev = var.split("@")
                var_url = f"{var_url_base}_pressure-level"
                urls.append(
                    f"{base_url}/{model_url}/{run_string}/{var_t}/{var_url}_{date_string}_{str(f_time).zfill(3)}_{plev}_{var_t.upper()}.grib2.bz2"
                )

    if invarient is not None:
        for var in invarient:
            var_url = f"{var_url_base}_time-invariant"
            urls.append(
                f"{base_url}/{model_url}/{run_string}/{var}/{var_url}_{date_string}_{var.upper()}.grib2.bz2"
            )
    return urls


def download_extract_files(urls: list, folder: str) -> list[str]:
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
        results = pool.map(download_extract_url, zip(urls_list, repeat(folder)))
        pool.close()
        pool.join()
    else:
        results = []
        for url in urls_list:
            results.append(download_extract_url((url, folder)))

    return results


def download_extract_url(url_and_folder):
    """
    Download and extract url if file isn't already downloaded

    Args:
        url_and_folder: Tuple of URL and folder

    Returns:

    """
    url, folder = url_and_folder
    filename = os.path.join(folder, os.path.basename(url).replace(".bz2", ""))

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


def get_dset(
    vars_2d=None,
    vars_3d=None,
    invarient=None,
    f_times=0,
    run="00",
    folder="/mnt/storage_ssd_4tb/DWD/",
    model="global",
):
    if vars_2d or vars_3d:
        date_string, _ = get_run(run)
        urls = find_file_name(
            vars_2d=vars_2d,
            vars_3d=vars_3d,
            invarient=invarient,
            f_times=f_times,
            model_url="icon/grib" if model == "global" else "icon-eu/grib",
            var_url_base="icon_global_icosahedral"
            if model == "global"
            else "icon-eu_europe_regular-lat-lon",
            run=run,
        )
        downloaded_files = download_extract_files(urls, folder)

    return downloaded_files
