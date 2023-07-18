#!/bin/bash

# Path to the Python script
SCRIPT_PATH="/home/zak/nwp/nwp/excarta/parse_excarta_monthly.py"

# Output directory for the Zarr files
OUTPUT_DIRECTORY="/mnt/storage_b/data/ocf/solar_pv_nowcasting/experimental/Excarta/sr_UK_Malta_full/zarr_format/r4/UK_excarta_"

# Iterate over the range of years
for year in {2018..2022}
do
    # Iterate over the range of months
    for month in {1..12}
    do
        echo "Processing data for ${year}-${month}..."
        python $SCRIPT_PATH $OUTPUT_DIRECTORY $year $month
    done
done
