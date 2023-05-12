#!/usr/bin/env bash

# Request access to the UK Met Office data on CEDA from this web page:
# https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f
#
# Then call this script with two arguments: <username> <password> <year>
# e.g.:
# ./download_UK_Met_Office_NWPs_from_CEDA.sh foo bar 2020
#
# Call this script in the directory into which you want to download data.
#
# The Met Office data on CEDA goes back to March 2016. This script
# will download about 4 terabytes per year of data.
#
# You probably want to run this script in a `gnu screen` session if you're
# SSH'ing into a VM or remote server.

wget --recursive -nH --cut-dirs=5 --no-clobber \
     --reject-regex ".*0300.*\.grib$" \
     --reject-regex ".*0900.*\.grib$" \
     --reject-regex ".*1500.*\.grib$" \
     --reject-regex ".*2100.*\.grib$" \
     --reject-regex ".*T120\.grib$" \
     --reject-regex ".*Wholesale[345].*\.grib$" \
     ftp://"$1":"$2"@ftp.ceda.ac.uk/badc/ukmo-nwp/data/ukv-grib/"$3"

# What are all those `--reject-regex` instructions doing?
#
# --reject-regex ".*[03|09|15|21]00_.*\.grib$"
#   rejects all NWPs initialised at 3, 9, 15, or 21 hours (and so you end up
#   with "only" four initialisations per day: 00, 06, 12, 18).
#
# --reject-regex ".*T120\.grib$"
#   rejects the `T120` files, which contain forecast steps from 2 days and
#   9 hours ahead, to 5 days ahead, in 3-hourly increments. So we accept the
#   `Wholesale[12].grib` files (steps from 00:00 to 1 day and 12 hours ahead,
#   in hourly increments) and `Wholesale[12]T54.grib` files (T54 runs from
#   1 day and 13 hours ahead to 2 days and 6 hours ahead. Hourly increments
#   from 1 day and 13 hours ahead to 2 days ahead. Then 3-hourly increments).
#
# --reject-regex ".*Wholesale[345].*\.grib$"
#   rejects the `Wholesale3`, 4, and 5 files. 5 are just static topography data,
#   so no need to download multiple copies of this data!
#
# Detailed docs of the Met Office data is available at:
# http://cedadocs.ceda.ac.uk/1334/1/uk_model_data_sheet_lores1.pdf
