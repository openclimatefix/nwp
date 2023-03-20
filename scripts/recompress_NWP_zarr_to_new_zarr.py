#!/usr/bin/env python
# coding: utf-8

"""Convert numerical weather predictions from the UK Met Office "UKV" model to Zarr.
"""
import datetime
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import xarray as xr
from numcodecs.bitround import BitRound
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
# Taken from page 4 of http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
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
    "--source_zarr_path",
    help="The input Zarr path to read from.",
)
@click.option(
    "--destination_zarr_path",
    help="The output Zarr path to write to.",
)
@click.option(
    "--step_chunk_size",
    default=1,
    type=int,
    help="The step chunk size",
)
@click.option(
    "--precision",
    default=32,
    type=int,
    help="The output precision, default is float32",
)
@click.option(
    "--bitround",
    default=None,
    help="The amount of bits to round. 11 bits for 99.99% of data, 9 bits for 99% of data",
)
@click.option(
    "--use_int8",
    default=False,
    help="Use int8 compression",
)
@click.option(
    "--log_level",
    default="DEBUG",
    type=click.Choice(_LOG_LEVELS),
    help="Optional.  Set the log level.",
)
@click.option(
    "--log_filename",
    default=None,
    help=(
        "Optional.  If not set then will default to `destination_zarr_path` with the"
        " suffix replaced with '.log'"
    ),
)
def main(
    source_zarr_path: str,
    destination_zarr_path: str,
    log_level: str,
    log_filename: Optional[str],
    use_int8: bool,
    precision: int,
    step_chunk_size: int,
    bitround: Optional[int],
):
    """The entry point into the script."""
    destination_zarr_path = Path(destination_zarr_path)

    # Set up logging.
    if log_filename is None:
        log_filename = destination_zarr_path.parent / (destination_zarr_path.stem + ".log")
    configure_logging(log_level=log_level, log_filename=log_filename)
    filter_eccodes_logging()

    # Open current Zarr
    source_dataset = xr.open_dataset(source_zarr_path, engine="zarr").sortby("init_time")

    # Rechunk
    source_dataset = source_dataset.chunk(
        {
            "init_time": 1,
            "step": step_chunk_size,
            "y": len(source_dataset.y) // 2,
            "x": len(source_dataset.x) // 2,
            "variable": -1,
        }
    )

    print(source_dataset)

    # options
    UKV_dict = {
        "compressor": Blosc2(cname="zstd", clevel=5),
    }

    if precision == 16:
        source_dataset["UKV"] = source_dataset.astype(np.float16)["UKV"]

    # Reset objects, see https://github.com/pydata/xarray/issues/3476
    for v in list(source_dataset.coords.keys()):
        if source_dataset.coords[v].dtype == object:
            source_dataset[v].encoding.clear()

    for v in list(source_dataset.variables.keys()):
        if source_dataset[v].dtype == object:
            source_dataset[v].encoding.clear()

    if bitround is not None:
        UKV_dict["filters"] = [BitRound(bitround)]  # 9 bits keeps 99% of the information

    to_zarr_kwargs = dict(
        encoding={
            "init_time": {"units": "nanoseconds since 1970-01-01"},
            "UKV": UKV_dict,
        },
    )

    logger.debug(f"Source Dataset Output: \n {source_dataset}")
    # The main event!
    source_dataset.to_zarr(destination_zarr_path, **to_zarr_kwargs, compute=True, mode="w")


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

    handlers = [logging.StreamHandler(), logging.FileHandler(log_filename, mode="a")]

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


def get_last_nwp_init_datetime_in_zarr(zarr_path: Path) -> datetime.datetime:
    """Get the last NWP init datetime in the Zarr."""
    dataset = xr.open_dataset(zarr_path, engine="zarr", mode="r")
    return dataset.init_time[-1].values


def dataset_has_variables(dataset: xr.Dataset) -> bool:
    """Return True if `dataset` has at least one variable."""
    return len(dataset.variables) > 0


if __name__ == "__main__":
    main()
