
import bz2
import os
import pandas as pd
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool, cpu_count

import requests

# If this works, then iterate through for now and download the raw data
dates = pd.date_range(start="2023-06-02", end="2023-07-03", freq="1D")
models = ["arpege-world", "arpege-europe", "arome-france"]
runs = ["00", "06", "12", "18"]
forecast_times_global = ["00H24H", "27H48H", "51H72H"]
forecast_times_europe = ["00H12H", "13H24H", "25H36H", "37H48H", "49H60H", "61H72H"]
forecast_times_france = ["00h06H", "07H12H", "13H18H", "19H24H", "25H30H", "31H36H",]
forecast_times = [forecast_times_global, forecast_times_europe, forecast_times_france]
pressure_levels = ["HP1", "HP2", "IP1", "IP2", "IP3", "IP4", "SP1", "SP2"]
for date in dates:
    for i, model in enumerate(models):
        for run in runs:
            for plevel in pressure_levels:
                for ftime in forecast_times[i]:
                    url = f"https://mf-nwp-models.s3.amazonaws.com/{model}/v1/{date.strftime('%Y-%m-%d')}/{run}/{plevel}/{ftime}.grib2"
                    filename = f"{model}_{date.strftime('%Y-%m-%d')}_{run}_{plevel}_{ftime}.grib2"
                    if os.path.exists(filename):
                        print(f"Skipping {url} as {filename} already exists")
                        continue
                    r = requests.get(url, stream=True)
                    if r.status_code == requests.codes.ok:
                        print(f"Downloading {url}")
                        with r.raw as source, open(filename, "wb") as dest:
                            dest.write((source.read()))
