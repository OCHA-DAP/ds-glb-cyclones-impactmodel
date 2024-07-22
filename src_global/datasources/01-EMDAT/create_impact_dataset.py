#!/usr/bin/env python3
import io
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src_global.utils import blob

PROJECT_PREFIX = "global_model"


def upload_in_chunks(
    dataframe,
    chunk_size,
    blob,
    blob_name_template,
    project_prefix=PROJECT_PREFIX,
):
    num_chunks = len(dataframe) // chunk_size + 1
    for i in range(num_chunks):
        chunk = dataframe[i * chunk_size : (i + 1) * chunk_size]
        if not chunk.empty:
            buffer = io.StringIO()
            chunk.to_csv(buffer, index=False)
            buffer.seek(0)
            chunk_blob_name = (
                f"{project_prefix}/EMDAT/{blob_name_template}_part_{i+1}.csv"
            )
            blob.upload_blob_data(
                blob_name=chunk_blob_name,
                data=buffer.getvalue(),
                prod_dev="dev",
            )
            print(f"Uploaded {chunk_blob_name}")


def clean_emdat():
    # Load emdat database and geolocated emdat
    impact_data = blob.load_csv(
        csv_path=f"{PROJECT_PREFIX}/EMDAT/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
    )
    geo_data = blob.load_csv(
        csv_path=f"{PROJECT_PREFIX}/EMDAT/pend-gdis-1960-2018-disasterlocations.csv"
    )

    impact_data_global = (
        impact_data[
            [
                "DisNo.",
                "Total Affected",
                "Start Year",
                "iso3",
                "Country",
                "Admin Units",
                "sid",
                "Event Name",
            ]
        ]
        .dropna(subset="Total Affected")
        .sort_values(["iso3", "sid", "Start Year"])
        .reset_index(drop=True)
    )

    # Clean & Merge
    impact_data_global["disasterno"] = impact_data_global["DisNo."].apply(
        lambda x: x[:-4]
    )
    impact_merged = geo_data[
        ["disasterno", "iso3", "latitude", "longitude"]
    ].merge(impact_data_global, on=["disasterno", "iso3"])
    return impact_merged


def geolocate_impact(impact_data):
    # Load shp
    global_shp = blob.load_gpkg(
        name=f"{PROJECT_PREFIX}/SHP/global_shapefile_GID_adm2.gpkg"
    )

    # Create geometry column
    geo_impact_data = gpd.GeoDataFrame(
        impact_data,
        geometry=gpd.points_from_xy(
            impact_data.longitude, impact_data.latitude
        ),
    )

    # Set the CRS for geo_impact_data if not already set
    geo_impact_data.set_crs(
        epsg=4326, inplace=True
    )  # Assuming WGS84 (epsg:4326)

    # Ensure both GeoDataFrames use the same CRS
    geo_impact_data.to_crs(global_shp.crs, inplace=True)

    # # Ensure both GeoDataFrames use the same CRS
    # geo_impact_data = geo_impact_data.to_crs(agg_df_shp.crs)

    # Add 2nd geometry column
    global_shp["shp_geometry"] = global_shp["geometry"]

    # Merge with shapefile
    merged_gdf = gpd.sjoin(
        geo_impact_data, global_shp, how="left", op="within"
    )

    # Identify points that were not matched
    unmatched_points = merged_gdf[merged_gdf["index_right"].isna()]

    # Find the nearest polygon for unmatched points
    unmatched_points = unmatched_points.drop(columns=["index_right"])
    global_shp["centroid"] = global_shp.centroid

    # Calculate distance to each polygon centroid and find the nearest
    nearest_polygons = gpd.GeoDataFrame(
        global_shp, geometry="centroid"
    ).sjoin_nearest(
        unmatched_points[geo_impact_data.columns],
        how="left",
        distance_col="distance",
    )
    # Keep the nearest point
    nearest_polygons = nearest_polygons.sort_values(
        "distance"
    ).drop_duplicates(subset="index_right", keep="first")
    # If the country does not match, dont consider the event. (it just affected 10 datapoints, quite good)
    nearest_polygons = nearest_polygons[
        nearest_polygons["GID_0"] == nearest_polygons["iso3"]
    ]
    nearest_polygons = nearest_polygons.drop(columns=["centroid", "distance"])

    # Merge back the geometry and other columns from agg_df_shp
    merged_gdf = merged_gdf.dropna(
        subset=["GID_0", "GID_1"]
    )  # .drop('centroid', axis=1)
    merged_gdf = pd.concat([merged_gdf, nearest_polygons], ignore_index=True)
    merged_gdf["geometry"] = merged_gdf["shp_geometry"]
    merged_gdf = merged_gdf.drop(["shp_geometry", "index_right"], axis=1)

    # Add level of impact
    merged_gdf["level"] = merged_gdf["Admin Units"].apply(
        lambda x: str(x)[3:7]
    )

    # Create "reduced" dataset with only relevant information
    reduced_impact_dataset = merged_gdf[
        [
            "Event Name",
            "DisNo.",
            "sid",
            "Total Affected",
            "level",
            "Start Year",
            "Country",
            "GID_0",
            "GID_1",
            "GID_2",
        ]
    ].drop_duplicates()
    reduced_shp = global_shp[["GID_0", "GID_1", "GID_2"]]

    # Iterate through every event to get non affected areas
    df_impact_complete = pd.DataFrame()
    for event in reduced_impact_dataset["DisNo."].unique():
        df_event = reduced_impact_dataset[
            reduced_impact_dataset["DisNo."] == event
        ]
        country = df_event.GID_0.unique()[0]
        level = df_event.level.unique()[0]
        df_loc = reduced_shp[reduced_shp.GID_0 == country]
        # Merge
        if level == "adm1":
            df_event = df_event.drop("GID_2", axis=1)
            reduced_merged = pd.merge(
                df_event, df_loc, on=["GID_0", "GID_1"], how="right"
            )
        elif level == "adm2":
            reduced_merged = pd.merge(
                df_event, df_loc, on=["GID_0", "GID_1", "GID_2"], how="right"
            )

        # Sort values for ffill
        reduced_merged = reduced_merged.sort_values(
            by="Total Affected", ascending=False
        )
        # Fill nans
        reduced_merged["DisNo."] = reduced_merged["DisNo."].ffill()
        reduced_merged["sid"] = reduced_merged["sid"].ffill()
        reduced_merged["Start Year"] = reduced_merged["Start Year"].ffill()
        reduced_merged["Event Name"] = reduced_merged["Event Name"].ffill()
        reduced_merged["level"] = reduced_merged["level"].ffill()
        reduced_merged["Country"] = reduced_merged["Country"].ffill()
        reduced_merged["Total Affected"].fillna(
            0, inplace=True
        )  # Fill 'Total Affected' with 0
        # Sort back by GID codes
        reduced_merged = reduced_merged.drop_duplicates().sort_values(
            ["GID_1", "GID_2"]
        )
        df_impact_complete = pd.concat([df_impact_complete, reduced_merged])

    # Save results

    chunk_size = 100000  # Adjust chunk size as necessary
    blob_name_template = "impact_data"
    upload_in_chunks(df_impact_complete, chunk_size, blob, blob_name_template)


if __name__ == "__main__":
    # Clean EMDAT dataset
    impact_data = clean_emdat()
    # Create impact dataset at adm2 level for event
    geolocate_impact(impact_data)
