#!/usr/bin/env python3
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src_global.utils import blob

PROJECT_PREFIX = "global_model"


def create_adm2_shp():
    # Load data
    global_shp = blob.load_gpkg(name=f"{PROJECT_PREFIX}/SHP/gadm_410-gpkg.zip")
    impact_data = blob.load_csv(
        csv_path=f"{PROJECT_PREFIX}/EMDAT/impact_data.csv"
    )
    iso3_list = impact_data.iso3.unique()

    # Process data
    global_shp_red = global_shp[
        ["GID_0", "GID_1", "GID_2", "GID_3", "GID_4", "GID_5", "geometry"]
    ]
    global_shp_red = global_shp_red[global_shp_red["GID_0"].isin(iso3_list)]

    # Group by the GID_2 level
    grouped = global_shp_red.groupby("GID_2")

    # Aggregate geometries using unary_union
    agg_geometries = grouped["geometry"].agg(lambda x: x.unary_union)

    # Create a new GeoDataFrame with the aggregated geometries
    agg_df = gpd.GeoDataFrame(agg_geometries, geometry="geometry")

    # Reset index to get GID_2 as a column again
    agg_df.reset_index(inplace=True)

    # Optionally, you can keep other relevant columns (e.g., GID_0, GID_1)
    # If you want to keep the first occurrence of these columns:
    agg_df = (
        grouped.first()
        .reset_index()[["GID_0", "GID_1", "GID_2"]]
        .merge(agg_df, on="GID_2")
    )

    # Save file
    agg_df_shp = gpd.GeoDataFrame(agg_df, geometry="geometry")

    datasets = {
        "/SHP/global_shapefile_GID_adm2.gpkg": agg_df_shp,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Save and upload GeoPackage datasets
        for filename, gdf in datasets.items():
            local_file_path = (
                temp_dir_path / filename.split("/")[-1]
            )  # Save in temp_dir with filename only
            gdf.to_file(local_file_path, driver="GPKG")
            blob_name = f"{PROJECT_PREFIX}{filename}"

            with open(local_file_path, "rb") as file:
                data = file.read()
                blob.upload_blob_data(
                    blob_name=blob_name, data=data, prod_dev="dev"
                )


if __name__ == "__main__":
    create_adm2_shp()
