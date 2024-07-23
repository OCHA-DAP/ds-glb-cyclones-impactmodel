import io
import os
import shutil
import sys
import tempfile
import zipfile
from io import BytesIO
from typing import Literal

import geopandas as gpd
import pandas as pd
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv

load_dotenv()

# Prod client (Not done)
DEV_BLOB_PROJ_URL = ""
prod_container_client = ""

# Dev client
DEV_BLOB_SAS = os.getenv("DEV_BLOB_SAS")

DEV_BLOB_BASE_URL = "https://imb0chd0dev.blob.core.windows.net/"
DEV_BLOB_PROJ_BASE_URL = DEV_BLOB_BASE_URL + "isi"
DEV_BLOB_PROJ_URL = DEV_BLOB_PROJ_BASE_URL + "?" + DEV_BLOB_SAS
dev_container_client = ContainerClient.from_container_url(DEV_BLOB_PROJ_URL)

PROJECT_PREFIX = "global_model"

def upload_gdf_to_blob(
    gdf, blob_name, prod_dev: Literal["prod", "dev"] = "dev"
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # File paths for shapefile components within the temp directory
        shp_base_path = os.path.join(temp_dir, "data")

        gdf.to_file(shp_base_path, driver="ESRI Shapefile")

        zip_file_path = os.path.join(temp_dir, "data")

        shutil.make_archive(
            base_name=zip_file_path, format="zip", root_dir=temp_dir
        )

        # Define the full path to the zip file
        full_zip_path = f"{zip_file_path}.zip"

        # Upload the buffer content as a blob
        with open(full_zip_path, "rb") as data:
            upload_blob_data(blob_name, data, prod_dev=prod_dev)


def load_gdf_from_blob(
    blob_name, shapefile: str = None, prod_dev: Literal["prod", "dev"] = "dev"
):
    blob_data = load_blob_data(blob_name, prod_dev=prod_dev)
    with zipfile.ZipFile(io.BytesIO(blob_data), "r") as zip_ref:
        zip_ref.extractall("temp")
        if shapefile is None:
            shapefile = [f for f in zip_ref.namelist() if f.endswith(".shp")][
                0
            ]
        gdf = gpd.read_file(f"temp/{shapefile}")
    return gdf


def load_blob_data(blob_name, prod_dev: Literal["prod", "dev"] = "dev"):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    return data


def upload_blob_data(
    blob_name, data, prod_dev: Literal["prod", "dev"] = "dev"
):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)


def list_container_blobs(
    name_starts_with=None, prod_dev: Literal["prod", "dev"] = "dev"
):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    return [
        blob.name
        for blob in container_client.list_blobs(
            name_starts_with=name_starts_with,
        )
    ]


def load_gpkg(name):
    return gpd.read_file(BytesIO(load_blob_data(name)))


def load_grid(complete=False):
    if complete:
        return gpd.read_file(
            BytesIO(
                load_blob_data(
                    PROJECT_PREFIX
                    + "/grid/output_dir/hti_0.1_degree_grid.gpkg"
                )
            )
        )
    else:
        return gpd.read_file(
            BytesIO(
                load_blob_data(
                    PROJECT_PREFIX
                    + "/grid/output_dir/hti_0.1_degree_grid_land_overlap.gpkg"
                )
            )
        )


def load_grid_centroids(complete=False):
    if complete:
        return gpd.read_file(
            BytesIO(
                load_blob_data(
                    PROJECT_PREFIX
                    + "/grid/output_dir/hti_0.1_degree_grid_centroids.gpkg"
                )
            )
        )
    else:
        return gpd.read_file(
            BytesIO(
                load_blob_data(
                    PROJECT_PREFIX
                    + "/grid/output_dir/hti_0.1_degree_grid_centroids_land_overlap.gpkg"
                )
            )
        )


def load_shp():
    return gpd.read_file(
        BytesIO(
            load_blob_data(
                PROJECT_PREFIX + "/SHP/global_shapefile_GID_adm2.gpkg"
            )
        )
    )


def load_emdat():
    return pd.read_csv(
        BytesIO(
            load_blob_data(PROJECT_PREFIX + "/EMDAT/impact_data_clean.csv")
        )
    )


def load_hti_distances():
    return pd.read_csv(
        BytesIO(
            load_blob_data(
                PROJECT_PREFIX + "/historical_forecasts/hti_distances.csv"
            )
        )
    )


def load_metadata():
    return pd.read_csv(
        BytesIO(
            load_blob_data(
                PROJECT_PREFIX + "/rainfall/input_dir/metadata_typhoons.csv"
            )
        )
    )


def load_csv(csv_path):
    return pd.read_csv(BytesIO(load_blob_data(csv_path)))


def upload_tif_to_blob(file_path, blob_path, prod_dev="dev"):
    # Read the TIFF file in binary mode
    with open(file_path, "rb") as file:
        tif_data = file.read()

    # Define the blob name where the file will be stored
    blob_name = f"{blob_path}/{file_path.name}"

    # Upload the TIFF data to the blob storage
    upload_blob_data(blob_name=blob_name, data=tif_data, prod_dev=prod_dev)


def delete_blob_data(blob_name, prod_dev: Literal["prod", "dev"] = "dev"):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.delete_blob()
