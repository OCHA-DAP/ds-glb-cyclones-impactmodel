#!/usr/bin/env python3
import io
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src_global.utils import blob

PROJECT_PREFIX = "global_model"


# Function for defining grid level for each country
def create_grid(shp, iso3):
    # Calculate the bounding box of the GeoDataFrame
    bounds = shp.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds

    # Define a margin to expand the bounding box (adjust as needed)
    margin = 1

    # Create expanded bounding box coordinates
    lon_min = np.round(lon_min - margin)
    lat_min = np.round(lat_min - margin)
    lon_max = np.round(lon_max + margin)
    lat_max = np.round(lat_max + margin)

    # Define grid
    xmin, xmax, ymin, ymax = (
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )  # Haiti extremes coordintates

    cell_size = 0.1
    cols = list(np.arange(xmin, xmax + cell_size, cell_size))
    rows = list(np.arange(ymin, ymax + cell_size, cell_size))
    rows.reverse()

    polygons = [
        Polygon(
            [
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y - cell_size),
                (x, y - cell_size),
            ]
        )
        for x in cols
        for y in rows
    ]
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=shp.crs)
    grid["id"] = grid.index + 1
    grid["iso3"] = iso3

    # %% Centroids
    # Extract lat and lon from the centerpoint
    grid["Longitude"] = grid["geometry"].centroid.map(lambda p: p.x)
    grid["Latitude"] = grid["geometry"].centroid.map(lambda p: p.y)
    grid["Centroid"] = (
        round(grid["Longitude"], 2).astype(str)
        + "W"
        + "_"
        + round(grid["Latitude"], 2).astype(str)
        + "N"
    )

    grid_centroids = grid.copy()
    grid_centroids["geometry"] = grid_centroids["geometry"].centroid

    # %% intersection of grid and shapefile
    adm2_grid_intersection = gpd.overlay(shp, grid, how="identity")
    adm2_grid_intersection = adm2_grid_intersection.dropna(subset=["id"])
    grid_land_overlap = grid.loc[grid["id"].isin(adm2_grid_intersection["id"])]

    # Centroids of intersection
    grid_land_overlap_centroids = grid_centroids.loc[
        grid["id"].isin(adm2_grid_intersection["id"])
    ]

    #  Grids by municipality
    grid_muni = gpd.sjoin(shp, grid_land_overlap, how="inner")

    intersection_areas = []
    for index, row in grid_muni.iterrows():
        id_cell = row["id"]
        grid_cell = grid_land_overlap[grid_land_overlap.id == id_cell].geometry
        municipality_polygon = row["geometry"]
        intersection_area = grid_cell.intersection(municipality_polygon).area
        intersection_areas.append(intersection_area)

    # Add area of intersection to each row
    grid_muni["intersection_area"] = [x.array[0] for x in intersection_areas]

    # Find the municipality with the largest intersection area
    # for each grid centroid and drop the rest
    grid_muni = grid_muni.sort_values("intersection_area", ascending=False)
    grid_muni_total = grid_muni.drop_duplicates(subset="id", keep="first")
    grid_muni_total = grid_muni_total[["id", "GID_0", "GID_1", "GID_2"]]

    # Change ID naming
    grid_muni_total["id"] = grid_muni_total["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid["id"] = grid["id"].apply(lambda x: iso3 + "_" + str(x))
    grid_land_overlap["id"] = grid_land_overlap["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid_centroids["id"] = grid_centroids["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid_land_overlap_centroids["id"] = grid_land_overlap_centroids[
        "id"
    ].apply(lambda x: iso3 + "_" + str(x))
    adm2_grid_intersection["id"] = adm2_grid_intersection["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )

    return (
        grid,
        grid_land_overlap,
        grid_centroids,
        grid_land_overlap_centroids,
        grid_muni_total,
    )


# For uploading large files to the blob
def upload_in_chunks(
    dataframe,
    chunk_size,
    blob,
    blob_name_template,
    project_prefix=PROJECT_PREFIX,
    folder="GRID",
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


def iterate_grid_creation():
    # Load global shapefile
    global_shp = blob.load_gpkg(
        name=f"{PROJECT_PREFIX}/SHP/global_shapefile_GID_adm2.gpkg"
    )
    # List of ISO3 regions
    isos = global_shp.GID_0.unique()

    # Iterate for every country
    grid_total = pd.DataFrame()
    grid_land_overlap_total = pd.DataFrame()
    grid_centroids_total = pd.DataFrame()
    grid_land_overlap_centroids_total = pd.DataFrame()
    grid_muni_total = pd.DataFrame()
    for iso3 in isos:
        shp = global_shp[global_shp.GID_0 == iso3]
        # Run function
        (
            grid,
            grid_land_overlap,
            grid_centroids,
            grid_land_overlap_centroids,
            grid_muni,
        ) = create_grid(shp=shp, iso3=iso3)
        # Append results
        grid_total = pd.concat([grid_total, grid])
        grid_land_overlap_total = pd.concat(
            [grid_land_overlap_total, grid_land_overlap]
        )
        grid_centroids_total = pd.concat(
            [grid_centroids_total, grid_centroids]
        )
        grid_land_overlap_centroids_total = pd.concat(
            [grid_land_overlap_centroids_total, grid_land_overlap_centroids]
        )
        grid_muni_total = pd.concat([grid_muni_total, grid_muni])

    # Reset index
    grid_total = grid_total.reset_index(drop=True)
    grid_land_overlap_total = grid_land_overlap_total.reset_index(drop=True)
    grid_centroids_total = grid_centroids_total.reset_index(drop=True)
    grid_land_overlap_centroids_total = (
        grid_land_overlap_centroids_total.reset_index(drop=True)
    )
    grid_muni_total = grid_muni_total.reset_index(drop=True)

    # Save datasets
    datasets = {
        "GRID/global_0.1_degree_grid.gpkg": grid_total,
        "GRID/global_0.1_degree_grid_centroids.gpkg": grid_centroids_total,
        "GRID/global_0.1_degree_grid_land_overlap.gpkg": grid_land_overlap_total,
        "GRID/global_0.1_degree_grid_centroids_land_overlap.gpkg": grid_land_overlap_centroids_total,
    }

    csv_datasets = {"GRID/grid_municipality_info.csv": grid_muni_total}

    # Create a temporary directory for saving files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Save and upload GeoPackage datasets
        for filename, gdf in datasets.items():
            local_file_path = temp_dir_path / filename.split("/")[-1]
            # Save in temp_dir with filename only
            gdf.to_file(local_file_path, driver="GPKG")
            blob_name = f"{PROJECT_PREFIX}/{filename}"

            with open(local_file_path, "rb") as file:
                data = file.read()
                blob.upload_blob_data(
                    blob_name=blob_name, data=data, prod_dev="dev"
                )

        # Save and upload CSV dataset (in chunks)
        chunk_size = 100000  # Adjust chunk size as necessary
        blob_name_template = "grid_municipality_info"
        upload_in_chunks(
            grid_muni_total,
            chunk_size,
            blob,
            blob_name_template,
            folder="GRID",
        )


if __name__ == "__main__":
    # Define grid level for every country available in the impact data
    iterate_grid_creation()
