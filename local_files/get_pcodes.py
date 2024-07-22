#!/usr/bin/env python3
from ast import literal_eval

import numpy as np
import pandas as pd

### For all events
import requests

# Load geo data and impact_Data
geo_data = pd.read_csv("pend-gdis-1960-2018-disasterlocations.csv")
impact_data_global = pd.read_csv("impact_data_global.csv")


# Clean & Merge
impact_data_global["disasterno"] = impact_data_global["DisNo."].apply(
    lambda x: x[:-4]
)
impact_merged = geo_data[["disasterno", "latitude", "longitude"]].merge(
    impact_data_global, on="disasterno"
)


def get_url(lat, lon, level):
    return f"https://apps.itos.uga.edu/CODV2API/api/v1/Themes/cod-ab/Lookup/latlng?latlong={lat},{lon}&wkid=4326&level={level}"


pcodes = []
for i, row in impact_merged.iterrows():
    try:
        lat = row["latitude"]
        lon = row["longitude"]
        level = 1  # we want adm1 information, for now.
        url = get_url(lat, lon, level)
        req = requests.get(url).text
        adm1_pcode = literal_eval(req[1:-1].replace("null", "None"))[
            "ADM1_PCODE"
        ]
        pcodes.append(adm1_pcode)
    except:
        pcodes.append(np.nan)


# Save it
impact_merged["ADM1_PCODE"] = pcodes
impact_merged.to_csv("impact_data_with_pcodes.csv", index=False)
