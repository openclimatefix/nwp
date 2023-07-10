#!/usr/bin/env python
# coding: utf-8

"""Convert numerical weather predictions from the UK Met Office "UKV" model to Zarr.

This script uses multiple processes to speed up the conversion. On leonardo (Open Climate Fix's
on-premises Threadripper 1920x server) this takes about 3 hours 15 minutes of processing per year
of NWP data (when using just Wholesale1 and Wholesale2 files).

Useful links:

* Met Office's data docs: https://zenodo.org/record/7357056

Some basic background to NWPs: The UK Met Office run their "UKV" model 8 times per day,
although, to save space on disk, we only use 4 of those runs per day (at 00, 06, 12, and 18 hours
past midnight UTC). The time the NWP model started running is called the "init_time".

Note that the UKV data is split into multiple files per NWP initialisation time.

How this script works:

1) The script finds all the .grib filenames specified by `source_path_and_search_pattern`.
2) Group the .grib filenames by the NWP initialisation datetime (which is the datetime given
   in the grib filename). For example, this groups together the Wholesale1 and Wholesale2 files
   for a specific init time.
3) If the `destination_zarr_path` exists then find the last NWP init time in the Zarr,
   and ignore all grib files with init times before that time. This allows the script to
   re-start where it left off if it crashes, or if new grib files are downloaded.
4) Use multiple worker processes. Each worker process is given a list of
   grib files associated with a single NWP init datetime. The worker process loads these grib
   files and does some simple post-processing, and then the worker process appends to the
   destination zarr.

The multi-processing aspect of the code is engineered to satisfy several constraints:
1) Only a single process can append to a Zarr store at once. And the appends must happen in strict
   order of the NWP initialisation time.
2) Passing large objects (such as a few hundred megabytes of NWP data) between processes via a
   multiprocessing.Queue is *really* slow: It takes about 7 seconds to pass an xr.Dataset
   representing Wholesale1 and Wholesale2 NWP data through a multiprocessing.Queue. This slowness
   is what motivates us to avoid passing the loaded Dataset between processes. Instead, we don't
   pass large objects between processes: Each xr.Dataset stays in one process. A single process
   loads the grib files associated with one init_time, processes, and writes the data to Zarr.

The code guarantees that the processes write to disk in order of the NWP init time by using
`multiprocessing.Pool.imap()`.

TROUBLESHOOTING
If you get any errors regarding .idx files then try deleting all *.idx files and trying again.

"""
import datetime
import glob
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Any, Optional, Union

import cfgrib
import click
import numpy as np
import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2

logger = logging.getLogger(__name__)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

NWP_VARIABLE_NAMES = (
    "cdcb",
    "lcc",
    "mcc",
    "hcc",
    "sde",
    "hcct",
    "dswrf",
    "dlwrf",
    "h",
    "t",
    "r",
    "dpt",
    "vis",
    "si10",
    "wdir10",
    "prmsl",
    "prate",
)

# Define geographical domain for UKV.
# Taken from page 4 of https://zenodo.org/record/7357056
# To quote the PDF:
#     "The United Kingdom domain is a 1,096km x 1,408km ~2km resolution grid."
DY_METERS = DX_METERS = 2_000
#     "The OS National Grid corners of the domain are:"
NORTH = 1223_000
SOUTH = -185_000
WEST = -239_000
EAST = 857_000
# Note that the UKV NWPs y is top-to-bottom, hence step is negative.
NORTHING = np.arange(start=NORTH, stop=SOUTH, step=-DY_METERS, dtype=np.int32)
EASTING = np.arange(start=WEST, stop=EAST, step=DX_METERS, dtype=np.int32)
NUM_ROWS = len(NORTHING)
NUM_COLS = len(EASTING)

# Define a set of grib variables to delete in load_grib_file().
VARS_TO_DELETE = (
    "unknown",
    "valid_time",
    "heightAboveGround",
    "heightAboveGroundLayer",
    "atmosphere",
    "cloudBase",
    "surface",
    "meanSea",
    "level",
)


@click.command()
@click.option(
    "--source_grib_path_and_search_pattern",
    default=(
        "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/NWP/"
        "UK_Met_Office/UKV/native/*/*/*/*Wholesale[12].grib"
    ),
    help=(
        "Optional. The directory and the search pattern for the source grib files."
        ' For example "/foo/bar/*/*/*/*Wholesale[12].grib". Should be in double quotes.'
    ),
    type=Path,
)
@click.option(
    "--destination_zarr_path",
    help="The output Zarr path to write to. Will be appended to if already exists.",
    type=Path,
)
@click.option(
    "--n_processes",
    default=8,
    help=(
        "Optional. Defaults to 8. The number of processes to use for loading grib"
        " files in parallel."
    ),
)
@click.option(
    "--log_level",
    default="DEBUG",
    type=click.Choice(_LOG_LEVELS),
    help="Optional. Set the log level.",
)
@click.option(
    "--log_filename",
    default=None,
    help=(
        "Optional. If not set then will default to `destination_zarr_path` with the"
        " suffix replaced with '.log'"
    ),
)
@click.option(
    "--n_grib_files_per_nwp_init_time",
    default=2,
    help=(
        "Optional. Defaults to 2. The number of grib files expected per NWP initialisation time."
        "  For example, if the search pattern includes Wholesale1 and Wholesale2 files, then set"
        " n_grib_files_per_nwp_init_time to 2."
    ),
)
def main(
    source_grib_path_and_search_pattern: Path,
    destination_zarr_path: Path,
    n_processes: int,
    log_level: str,
    log_filename: Optional[str],
    n_grib_files_per_nwp_init_time: int,
):
    """The entry point into the script."""
    # Set up logging.
    if log_filename is None:
        log_filename = destination_zarr_path.parent / (destination_zarr_path.stem + ".log")
    configure_logging(log_level=log_level, log_filename=log_filename)
    filter_eccodes_logging()

    # Get all filenames
    source_grib_path_and_search_pattern = source_grib_path_and_search_pattern.expanduser()
    logger.info(f"Getting list of all filenames in {source_grib_path_and_search_pattern}...")
    filenames = glob.glob(str(source_grib_path_and_search_pattern))
    filenames = [Path(filename) for filename in filenames]
    logger.info(f"Found {len(filenames):,d} grib filenames.")
    if len(filenames) == 0:
        logger.warning(
            "No files found! Are you sure the source_grib_path_and_search_pattern is correct?"
        )
        return

    # Decode and group the grib filenames:
    map_datetime_to_grib_filename = decode_and_group_grib_filenames(
        filenames=filenames, n_grib_files_per_nwp_init_time=n_grib_files_per_nwp_init_time
    )

    # Remove grib filenames which have already been processed:
    map_datetime_to_grib_filename = select_grib_filenames_still_to_process(
        map_datetime_to_grib_filename, destination_zarr_path
    )

    # The main event!
    process_grib_files_in_parallel(
        map_datetime_to_grib_filename=map_datetime_to_grib_filename,
        destination_zarr_path=destination_zarr_path,
        n_processes=n_processes,
    )


def configure_logging(log_level: str, log_filename: str) -> None:
    """Configure logger for this script.

    Args:
      log_level: String like "DEBUG".
      log_filename: The full filename of the log file.
    """
    assert log_level in _LOG_LEVELS
    log_level = getattr(logging, log_level)  # Convert string to int.
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s processID=%(process)d %(message)s")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode="a"),
    ]

    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def filter_eccodes_logging():
    """Filter out "ecCodes provides no latitudes/longitudes for gridType='transverse_mercator'"

    Filter out this warning because it is not useful, and just adds noise to the log.
    """

    # The warning originates from here:
    # https://github.com/ecmwf/cfgrib/blob/master/cfgrib/dataset.py#L402
    class FilterEccodesWarning(logging.Filter):
        def filter(self, record) -> bool:
            """Inspect `record`. Return True to log `record`. Return False to ignore `record`."""
            return not record.getMessage() == (
                "ecCodes provides no latitudes/longitudes for gridType='transverse_mercator'"
            )

    logging.getLogger("cfgrib.dataset").addFilter(FilterEccodesWarning())


def grib_filename_to_datetime(full_grib_filename: Path) -> datetime.datetime:
    """Parse the grib filename and return the datetime encoded in the filename.

    Returns a datetime.
      For example, if the filename is '/foo/202101010000_u1096_ng_umqv_Wholesale1.grib',
      then the returned datetime will be datetime(year=2021, month=1, day=1, hour=0, minute=0).

    Raises RuntimeError if the filename does not match the expected regex pattern.
    """
    # Get the base_filename, which will be of the form '202101010000_u1096_ng_umqv_Wholesale1.grib'
    base_filename = full_grib_filename.name

    # Use regex to match the year, month, day, hour, and minute.
    # That is, group the filename as shown in these brackets:
    #   (2021)(01)(01)(00)(00)_u1096_ng_umqv_Wholesale1.grib
    # A quick guide to the relevant regex operators:
    #   ^ matches the beginning of the string.
    #   () defines a group.
    #   (?P<name>...) names the group. We can access the group with regex_match.groupdict()[<name>]
    #   \d matches a single digit.
    #   {n} matches the preceding item n times.
    #   . matches any character.
    #   $ matches the end of the string.
    regex_pattern_string = (
        "^"  # Match the beginning of the string.
        "(?P<year>\d{4})"  # noqa: W605
        "(?P<month>\d{2})"  # noqa: W605
        "(?P<day>\d{2})"  # noqa: W605
        "(?P<hour>\d{2})"  # noqa: W605
        "(?P<minute>\d{2})"  # noqa: W605
        "_u1096_ng_umqv_Wholesale\d\.grib$"  # noqa: W605. Match the end of the string.
    )
    regex_pattern = re.compile(regex_pattern_string)
    regex_match = regex_pattern.match(base_filename)
    if regex_match is None:
        msg = (
            f"Filename '{full_grib_filename}' does not conform to expected"
            f" regex pattern '{regex_pattern_string}'!"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    # Convert strings to ints:
    regex_groups: dict[str, int] = {
        key: int(value) for key, value in regex_match.groupdict().items()
    }

    return datetime.datetime(**regex_groups)


def decode_and_group_grib_filenames(
    filenames: list[Path], n_grib_files_per_nwp_init_time: int = 2
) -> pd.Series:
    """Returns a pd.Series where the index is the NWP init time.

    And the values are the full_grib_filename of each grib file.

    Throws away any groups where there are not exactly n_grib_files_per_nwp_init_time.
    """
    # Create a 1D array of NWP initialisation times (decoded from the grib filenames):
    n_filenames = len(filenames)
    nwp_init_datetimes = np.full(shape=n_filenames, fill_value=np.NaN, dtype="datetime64[ns]")
    for i, filename in enumerate(filenames):
        nwp_init_datetimes[i] = grib_filename_to_datetime(filename)

    # Create a pd.Series where each row maps an NWP init datetime to the full grib filename:
    map_datetime_to_filename = pd.Series(
        filenames, index=nwp_init_datetimes, name="full_grib_filename"
    )
    del nwp_init_datetimes
    map_datetime_to_filename.index.name = "nwp_init_datetime_utc"

    # Select only rows where there are exactly n_grib_files_per_nwp_init_time:
    def _filter_func(group):
        return group.count() == n_grib_files_per_nwp_init_time

    map_datetime_to_filename = map_datetime_to_filename.groupby(level=0).filter(_filter_func)

    return map_datetime_to_filename.sort_index()


def select_grib_filenames_still_to_process(
    map_datetime_to_grib_filename: pd.Series, destination_zarr_path: Path
) -> pd.Series:
    """Remove grib filenames for NWP init times that already exist in Zarr."""
    if destination_zarr_path.exists():
        last_nwp_init_datetime_in_zarr = get_last_nwp_init_datetime_in_zarr(destination_zarr_path)
        logger.info(
            f"{destination_zarr_path} exists. The last NWP init datetime (UTC) in the Zarr is"
            f" {last_nwp_init_datetime_in_zarr}"
        )
        nwp_init_datetimes_utc = map_datetime_to_grib_filename.index
        map_datetime_to_grib_filename = map_datetime_to_grib_filename[
            nwp_init_datetimes_utc > last_nwp_init_datetime_in_zarr
        ]
    return map_datetime_to_grib_filename


def get_last_nwp_init_datetime_in_zarr(zarr_path: Path) -> datetime.datetime:
    """Get the last NWP init datetime in the Zarr."""
    dataset = xr.open_dataset(zarr_path, engine="zarr", mode="r")
    return dataset.init_time[-1].values


def load_grib_file(full_grib_filename: Union[Path, str]) -> xr.Dataset:
    """Merges and loads all contiguous xr.Datasets for a single grib file.

    Removes unnecessary variables. Picks heightAboveGround = 1 meter for temperature.

    Returns an xr.Dataset which has been loaded from disk. Loading from disk at this point
    takes about 2 seconds for a 250 MB grib file, but speeds up reshape_1d_to_2d
    from about 7 seconds to 0.5 seconds :)

    Args:
      full_grib_filename:  The full filename (including the path) of a single grib file.
    """
    # The grib files are "heterogeneous" so we cannot use xr.open_dataset().
    # Instead we use cfgrib.open_datasets() which returns a *list* of contiguous xr.Datasets.
    # See https://github.com/ecmwf/cfgrib#automatic-filtering
    logger.debug(f"Opening {full_grib_filename}...")
    datasets_from_grib: list[xr.Dataset] = cfgrib.open_datasets(full_grib_filename)
    n_datasets = len(datasets_from_grib)

    # Get each dataset into the right shape for merging:
    # We use `for i in range(n_datasets)` instead of `for ds in datasets_from_grib`
    # because the loop modifies each dataset:
    for i in range(n_datasets):
        ds = datasets_from_grib[i]

        logger.debug(f"Dataset {i} before processing:\n{ds}\n")

        # We want the temperature at 1 meter above ground, not at 0 meters above ground.
        # In the early NWPs (definitely in the 2016-03-22 NWPs), `heightAboveGround` only has
        # 1 entry ("1" meter above ground) and `heightAboveGround` isn't set as a dimension for `t`.
        # In later NWPs, 'heightAboveGround' has 2 values (0, 1) and is a dimension for `t`.
        if "t" in ds and "heightAboveGround" in ds["t"].dims:
            ds = ds.sel(heightAboveGround=1)

        # Delete unnecessary variables.
        for var_name in VARS_TO_DELETE:
            try:
                del ds[var_name]
            except KeyError as e:
                logger.debug(f"var name not in dataset: {e}")
            else:
                logger.debug(f"Deleted {var_name}")

        logger.debug(f"Dataset {i} after processing:\n{ds}\n")
        logger.debug("**************************************************")

        datasets_from_grib[i] = ds
        del ds  # Save memory.

    merged_ds = xr.merge(datasets_from_grib)
    del datasets_from_grib  # Save memory.
    logger.debug(f"Loading {full_grib_filename}...")
    return merged_ds.load()


def reshape_1d_to_2d(dataset: xr.Dataset) -> xr.Dataset:
    """Convert 1D into 2D array.

    In the grib files, the pixel values are in a flat 1D array (indexed by the `values` dimension).
    The ordering of the pixels in the grib are left to right, bottom to top.

    This function replaces the `values` dimension with an `x` and `y` dimension,
    and, for each step, reshapes the images to be 2D.
    """
    # Adapted from https://stackoverflow.com/a/62667154

    # Don't reshape yet. Instead just create new coordinates,
    # which give the `x` and `y` position for each position in the `values` dimension:
    dataset = dataset.assign_coords(
        {
            "x": ("values", np.tile(EASTING, reps=NUM_ROWS)),
            "y": ("values", np.repeat(NORTHING, repeats=NUM_COLS)),
        }
    )

    # Now set `values` to be a MultiIndex, indexed by `y` and `x`:
    dataset = dataset.set_index(values=("y", "x"))

    # Now unstack. This gets rid of the `values` dimension and indexes
    # the data variables using `y` and `x`.
    return dataset.unstack("values")


def dataset_has_variables(dataset: xr.Dataset) -> bool:
    """Return True if `dataset` has at least one variable."""
    return len(dataset.variables) > 0


def post_process_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Get the Dataset ready for saving to Zarr.

    Convert the Dataset (with different DataArrays for each NWP variable)
    to a single DataArray with a `variable` dimension. We do this so each
    Zarr chunk can hold multiple NWP variables (which is useful because
    we often load all the NWP variables at once).

    Rename `time` to `init_time` (because `time` is ambiguous. NWPs have two "times":
    the initialisation time and the target time).
    """
    logger.debug("Post-processing dataset...")
    da = dataset.to_array(dim="variable", name="UKV")

    assert len(da["variables"]) <= len(NWP_VARIABLE_NAMES)
    # to_array looks like it can sometimes change the order of the variables.
    # So fix the order:
    # If some variables are missing, then reindexing will simply set those as NaN.
    if (
        len(da["variable"]) != len(NWP_VARIABLE_NAMES)
        or not (da["variable"] == NWP_VARIABLE_NAMES).all()
    ):
        logger.warning("Fixing the order of the NWP variable names.")
        da = da.reindex(variable=list(NWP_VARIABLE_NAMES))

    # Reverse `y` so it's top-to-bottom (so ZarrDataSource.get_example() works correctly!)
    # Adapted from:
    # https://stackoverflow.com/questions/54677161/xarray-reverse-an-array-along-one-coordinate
    y_reversed = da.y[::-1]
    da = da.reindex(y=y_reversed)

    da = da.astype(np.float16)

    return da.to_dataset().rename({"time": "init_time"})


def append_to_zarr(dataset: xr.Dataset, zarr_path: Union[str, Path]):
    """If zarr_path already exists then append to the init_time dim. Else create a new Zarr.

    If creating a new Zarr, then this function sets the units for representing time to
    "nanoseconds since 1970-01-01" (which is the encoding used by `numpy.datetime64[ns]`) otherwise,
    by default, xarray defaults representing time as an integer numbers of *days* and hence cannot
    represent sub-day temporal resolution and corrupts the `init_time` values when we
    append to Zarr. See:
        https://github.com/pydata/xarray/issues/5969   and
        http://xarray.pydata.org/en/stable/user-guide/io.html#time-units

    Also sets the compressor to `Blosc2(cname="zstd", clevel=5)` which has been shown
    to provide a good balance of speed and small file sizes in empirical testing.
    """
    zarr_path = Path(zarr_path)
    if zarr_path.exists():
        # Append to existing Zarr store.
        to_zarr_kwargs = dict(
            append_dim="init_time",
        )

        # Check that dataset has same dimensions as the dataset on disk:
        assert len(dataset["step"]) == 37
    else:
        # Create new Zarr store.
        chunk_shapes_dict = {
            "variable": -1,
            "init_time": 1,
            "step": -1,
            "y": len(dataset.y) // 2,
            "x": len(dataset.x) // 2,
        }
        assert set(chunk_shapes_dict.keys()) == set(dataset["UKV"].dims)
        chunk_shapes_list = [chunk_shapes_dict[dim_name] for dim_name in dataset["UKV"].dims]
        to_zarr_kwargs = dict(
            encoding={
                "init_time": {"units": "nanoseconds since 1970-01-01"},
                "UKV": {
                    "compressor": Blosc2(cname="zstd", clevel=5),
                    "chunks": chunk_shapes_list,
                },
            },
        )
    dataset.to_zarr(zarr_path, **to_zarr_kwargs)


def load_grib_files_for_single_nwp_init_time(
    full_grib_filenames: list[Path], task_number: int
) -> Optional[xr.Dataset]:
    """Returns processed Dataset merging all grib files specified by full_grib_filenames.

    Returns None if any of the grib files are invalid.
    """
    assert len(full_grib_filenames) > 0
    datasets_for_nwp_init_datetime = []
    for full_grib_filename in full_grib_filenames:
        logger.debug(f"Task #{task_number}: Opening {full_grib_filename}")
        try:
            dataset_for_filename = load_grib_file(full_grib_filename)
        except EOFError as e:
            logger.warning(f"{e}. Filesize = {full_grib_filename.stat().st_size:,d} bytes")
            # If any of the files associated with this nwp_init_datetime is broken then
            # skip all, because we don't want incomplete data for an init_datetime.
            return None
        else:
            if dataset_has_variables(dataset_for_filename):
                datasets_for_nwp_init_datetime.append(dataset_for_filename)
            else:
                logger.warning(f"{full_grib_filename} has no variables!")
                return None
    logger.debug(f"Task #{task_number}: Merging datasets...")
    dataset_for_nwp_init_datetime = xr.merge(datasets_for_nwp_init_datetime)
    del datasets_for_nwp_init_datetime  # Save memory.
    logger.debug(f"Task #{task_number}: Reshaping datasets...")
    dataset_for_nwp_init_datetime = reshape_1d_to_2d(dataset_for_nwp_init_datetime)
    dataset_for_nwp_init_datetime = dataset_for_nwp_init_datetime.expand_dims("time", axis=0)
    dataset_for_nwp_init_datetime = post_process_dataset(dataset_for_nwp_init_datetime)
    return dataset_for_nwp_init_datetime


def load_grib_files_for_nwp_init_time_with_exception_logging_and_timing(
    task: dict[str, Any]
) -> Optional[xr.Dataset]:
    """Wrapper around load_grib_files_for_single_nwp_init_time with exception logging & timing.

    Args:
        task: A dict which contains the arguments for loading grib files. We use a dict
              because multiprocessing.Pool.imap() requires a single iterable (and we use
              a single iterable of dicts). Task must contains these keys:
              - row:  The pd.Series listing the full grib filenames to load.
              - task_number: The number of this task. Used for logging.
              - destination_zarr_path: The path of the Zarr to append to.
    """
    full_grib_filenames = task["row"]
    task_number = task["task_number"]

    start_time = pd.Timestamp.now()

    logger.debug(f"Task #{task_number} starting...")
    try:
        dataset = load_grib_files_for_single_nwp_init_time(full_grib_filenames, task_number)
    except Exception:
        logger.exception(
            f"Exception raised when processing task number {task_number},"
            f" loading grib filenames {full_grib_filenames}"
        )
        raise

    if dataset is None:
        logger.warning(
            f"Task #{task_number}: Dataset is None! Grib filenames = {full_grib_filenames}"
        )
        return None

    # Compute the time taken for this NWP init time.
    time_taken = pd.Timestamp.now() - start_time
    logger.debug(
        f"Task #{task_number}: Finished loading NWP init time {dataset.init_time.values}"
        f" in {time_taken}"
    )
    return dataset


def process_grib_files_in_parallel(
    map_datetime_to_grib_filename: pd.Series,
    destination_zarr_path: Path,
    n_processes: int,
) -> None:
    """Process grib files in parallel."""
    start_time = pd.Timestamp.now()

    # Create a list of `tasks` which include the grib filenames:
    tasks: list[dict[str, object]] = []
    for task_number, (_, row) in enumerate(map_datetime_to_grib_filename.groupby(level=0)):
        tasks.append(
            dict(
                row=row,  # The pd.Series listing the full grib filenames to load.
                task_number=task_number,
                destination_zarr_path=destination_zarr_path,
            )
        )

    logger.info(f"About to process {len(tasks):,d} tasks using {n_processes} processes.")

    # Run the processes!
    with multiprocessing.Pool(processes=n_processes) as pool:
        for ds in pool.imap(
            load_grib_files_for_nwp_init_time_with_exception_logging_and_timing,
            tasks,
        ):
            if ds is not None:
                append_to_zarr(ds, destination_zarr_path)

    # Compute the time taken in total.
    time_taken = pd.Timestamp.now() - start_time
    seconds_per_task = (time_taken / (len(tasks) + 1)).total_seconds()
    logger.info(
        f"{len(tasks):,d} tasks (NWP init timesteps) completed in {time_taken}"
        f". That's an average of {seconds_per_task:,.1f} seconds per NWP init timestep."
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
