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
from rasterio.io import MemoryFile

load_dotenv()

# Dev client for monitoring
DEV_BLOB_SAS_GLOBAL = os.getenv("DEV_BLOB_SAS_GLOBAL")

DEV_BLOB_BASE_GLOBAL_URL = "https://imb0chd0dev.blob.core.windows.net/"
DEV_BLOB_PROJ_BASE_GLOBAL_URL = DEV_BLOB_BASE_GLOBAL_URL + "global"
DEV_BLOB_PROJ_GLOBAL_URL = (
    DEV_BLOB_PROJ_BASE_GLOBAL_URL + "?" + DEV_BLOB_SAS_GLOBAL
)
prod_container_client = ContainerClient.from_container_url(
    DEV_BLOB_PROJ_GLOBAL_URL
)

# Dev client
DEV_BLOB_SAS = os.getenv("DEV_BLOB_SAS")

DEV_BLOB_BASE_URL = "https://imb0chd0dev.blob.core.windows.net/"
DEV_BLOB_PROJ_BASE_URL = DEV_BLOB_BASE_URL + "isi"
DEV_BLOB_PROJ_URL = DEV_BLOB_PROJ_BASE_URL + "?" + DEV_BLOB_SAS
dev_container_client = ContainerClient.from_container_url(DEV_BLOB_PROJ_URL)

PROJECT_PREFIX = "global_model"


# To load data
def load_blob_data(blob_name, prod_dev: Literal["prod", "dev"] = "dev"):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    return data


# To upload data
def upload_blob_data(
    blob_name, data, prod_dev: Literal["prod", "dev"] = "dev"
):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data, overwrite=True)


# List all files into blob
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


# For loading gpkg files
def load_gpkg(name):
    return gpd.read_file(BytesIO(load_blob_data(name)))


# For loading csv files
def load_csv(csv_path):
    return pd.read_csv(BytesIO(load_blob_data(csv_path)))


# Function to upload a tif file to blob
def upload_tif_to_blob(file_path, blob_name):
    with open(file_path, "rb") as file:
        data = file.read()
        upload_blob_data(blob_name=blob_name, data=data)


# Function to load tif from blob
def load_tif_from_blob(blob_name, prod_dev: Literal["prod", "dev"] = "dev"):
    data = load_blob_data(blob_name, prod_dev=prod_dev)

    # Check if data is retrieved correctly
    if not data:
        raise ValueError("No data retrieved from blob")

    # Create an in-memory file with MemoryFile
    memfile = MemoryFile(data)

    # Open the dataset from the in-memory file
    dataset = memfile.open()

    # Perform operations with the dataset
    print(dataset.profile)  # Example: print the metadata
    return dataset


# For deleting blob files
def delete_blob_data(blob_name, prod_dev: Literal["prod", "dev"] = "dev"):
    if prod_dev == "dev":
        container_client = dev_container_client
    else:
        container_client = prod_container_client
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.delete_blob()


# For uploading to blob in chunks (for large files)
def upload_in_chunks(
    dataframe,
    chunk_size,
    blob,
    blob_name_template,
    folder,
    project_prefix="global_model",
):
    num_chunks = len(dataframe) // chunk_size + 1
    for i in range(num_chunks):
        chunk = dataframe[i * chunk_size : (i + 1) * chunk_size]
        if not chunk.empty:
            buffer = io.StringIO()
            chunk.to_csv(buffer, index=False)
            buffer.seek(0)
            chunk_blob_name = f"{project_prefix}/{folder}/{blob_name_template}_part_{i+1}.csv"
            blob.upload_blob_data(
                blob_name=chunk_blob_name,
                data=buffer.getvalue(),
                prod_dev="dev",
            )
            print(f"Uploaded {chunk_blob_name}")


# Functions to load specific datasets
def get_municipality_info():
    # Load municipality info datasets (is in chunks)
    filenames = [f"grid_municipality_info_part_{i}.csv" for i in range(1, 9)]
    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/GRID/{filename}"
        df = load_csv(csv_path=csv_path)
        dataframes.append(df)
    global_ids_mun = pd.concat(dataframes, ignore_index=True)
    return global_ids_mun


def get_population_data():
    # Load population dataset (is in chunks)
    filenames = [f"pop_grid_global_part_{i}.csv" for i in range(1, 9)]
    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/WORLDPOP/processed_pop/{filename}"
        df = load_csv(csv_path=csv_path)
        dataframes.append(df)
    global_pop = pd.concat(dataframes, ignore_index=True)
    return global_pop


def get_impact_data():
    # Load impact data (is in chunks)
    filenames = [f"impact_data_part_{i}.csv" for i in range(1, 9)]
    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/EMDAT/{filename}"
        df = load_csv(csv_path=csv_path)
        dataframes.append(df)
    impact_global = pd.concat(dataframes, ignore_index=True)
    return impact_global


def get_impact_data_at_grid_level(weather_constraints=False):
    if weather_constraints:
        # Add here this option later
        filenames = []
    elif weather_constraints == False:
        # Load impact data (is in chunks)
        filenames = [
            f"impact_data_grid_global_part_{i}.csv" for i in range(1, 184)
        ]

    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/EMDAT/grid_based/{filename}"
        df = load_csv(csv_path=csv_path)
        dataframes.append(df)
    impact_global = pd.concat(dataframes, ignore_index=True)
    return impact_global
