# NWP
Tools for downloading and processing numerical weather predictions

At the moment, this code is focused on downloading historical UKV NWPs produced by the UK Met Office, and distributed by CEDA.

Detailed docs of the Met Office data is available [here](http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf).

# Installation

## `conda`

From within the cloned `nwp` directory:

```shell
conda env create -f environment.yml
conda activate nwp
pip install -e .
pre-commit install
```

# Downloading UKV numerical weather predictions from CEDA

Request access to the [UK Met Office data on CEDA](https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f).

Once you have a username and password, download using
[`scripts/download_UK_Met_Office_NWPs_from_CEDA.sh`](https://github.com/openclimatefix/nwp/tree/main/scripts/download_UK_Met_Office_NWPs_from_CEDA.sh).
Please see the comments at the top of the script for instructions.

# Convert grib files to Zarr

Then convert the `grib` files to Zarr using `scripts/convert_NWP_grib_to_zarr.py`. Run that script
with `--help` to see how to operate it. See the comments at the top of the script to learn how
the script works.
