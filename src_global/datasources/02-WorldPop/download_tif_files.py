#!/usr/bin/env python3
import tempfile
from pathlib import Path

from wpgpDownload.utils.convenience_functions import (
    download_country_covariates as dl,
)

from src_global.utils import blob


def download_worldpop_tif_files(iso3_list):
    # Iterate and download tif files for every country
    i = 0
    for iso3 in iso3_list[1:2]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # In a temporary file
            temp_path = Path(temp_dir)

            # Download tif file
            print(f"Downloading {iso3}, {i}/{len(iso3_list)}")
            dl(ISO=iso3, out_folder=temp_path, prod_name=["ppp_2020_UNadj"])

            # Define path to upload to blob
            file_path = temp_path / f"{iso3.lower()}_ppp_2020_UNadj.tif"
            blob_path = f"global_model/WORLDPOP"

            # Check if the file is already in the blob storage
            full_path = f"{blob_path}/{iso3.lower()}_ppp_2020_UNadj.tif"
            check = len(blob.list_container_blobs(name_starts_with=full_path))

            if check == 1:
                print(f"{iso3} already in the blob")
                pass
            else:
                blob.upload_tif_to_blob(file_path, blob_path)
                i += 1


if __name__ == "__main__":
    # Define countries to download pop data
    iso3_list = [
        "BGD",
        "HTI",
        "MDG",
        "MMR",
        "MOZ",
        "CHN",
        "AUS",
        "BRA",
        "COL",
        "CRI",
        "CUB",
        "SLV",
        "FJI",
        "GTM",
        "IND",
        "IRN",
        "PRK",
        "KOR",
        "MEX",
        "NIC",
        "PAK",
        "PHL",
        "SOM",
        "TWN",
        "THA",
        "USA",
        "VNM",
        "DOM",
        "JPN",
        "PAN",
        "YEM",
        "BHS",
        "BRB",
        "BLZ",
        "COM",
        "DMA",
        "HND",
        "IDN",
        "JAM",
        "LAO",
        "MUS",
        "OMN",
        "PNG",
        "RUS",
        "SLB",
        "LKA",
        "TON",
        "TTO",
        "VUT",
        "COD",
        "CAN",
        "NZL",
    ]
    # Download tif files for every country
    download_worldpop_tif_files(iso3_list=iso3_list)