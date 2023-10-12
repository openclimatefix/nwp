import xarray as xr
import ocf_blosc2
import os
import tempfile
import zipfile
import click


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.argument("year", type=str)
@click.argument("output", type=click.Path(), metavar="Output Directory")
def ecmwf_merge(path, year, output):
    """
    Merge multiple ECMWF datasets into a single dataset.

    Args:
        path (str): Path to directory containing the input datasets.
        year (str): Year of the datasets to merge.
        output (str): Path to output directory where the merged dataset will be saved.
                      The filename will be generated automatically using the year.

    Returns:
        merged_ds (xarray.Dataset): Merged dataset containing all input datasets.

    Example:
        $ python process_raw_ecmwf.py /path/to/datasets 2021 /path/to/output/
    """
    zarr_files = [
        os.path.join(path, file)
        for file in os.listdir(path)
        if file.endswith(".zarr.zip") & file.startswith(f"{year}")
    ]
    datasets = [xr.open_zarr(file) for file in zarr_files]

    # Convert all of the values in the `variable` variable to `str` before merging the datasets.
    for dataset in datasets:
        dataset["variable"] = dataset["variable"].astype(str)

    print("Merging data")
    merged_ds = xr.concat(datasets, dim="init_time")
    merged_ds = merged_ds.sortby("init_time")
    print("Saving data")

    output = str(os.path.join(output, year + ".zarr"))
    print(output)

    merged_ds["latitude"] = merged_ds["latitude"].astype(float)
    merged_ds["longitude"] = merged_ds["longitude"].astype(float)

    merged_ds.to_zarr(output, mode="w")
    print(f"Saved Zarr to {output}")
    return merged_ds


if __name__ == "__main__":
    ecmwf_merge()
