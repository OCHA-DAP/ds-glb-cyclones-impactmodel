#!/usr/bin/env python3
import datetime as dt
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
# import rasterio
import rioxarray as rxr
# from rasterstats import zonal_stats

from src_global.utils import blob, constant


# Function to download GPM data from blob storage
def get_raster_files(df_meta, sid, DAYS_TO_LANDFALL=2):
    metadata = df_meta[df_meta.sid == sid]
    start_date = metadata["landfalldate"] - dt.timedelta(days=DAYS_TO_LANDFALL)
    end_date = metadata["landfalldate"] + dt.timedelta(days=DAYS_TO_LANDFALL)
    date_list = pd.date_range(start_date.iloc[0], end_date.iloc[0])

    raster_files = []
    for date in date_list:
        date = date.strftime("%Y-%m-%d")
        print(date)
        blob_name = f"imerg/v6/imerg-daily-late-{str(date)}.tif"
        raster_file = blob.load_tif_from_blob(
            blob_name=blob_name, prod_dev="prod"
        )
        raster_files.append(raster_file)

    return date_list, raster_files


# PPS data to grid
def create_rainfall_dataset(grid_global, df_meta, iso3, sid):
    # Get rasters
    date_list, raster_files = get_raster_files(
        df_meta=df_meta, sid=sid, DAYS_TO_LANDFALL=2
    )

    # Call grid and raster files
    grid = grid_global[grid_global.iso3 == iso3]
    grid = gpd.GeoDataFrame(grid, geometry="geometry")
    # Convert grid cells to GeoDataFrame with bounding boxes
    grid["bbox"] = grid.geometry.apply(lambda geom: geom.bounds)

    # For every raster file (date)
    file_df = pd.DataFrame()
    i = 0
    for src in raster_files:
        # Load the raster data
        da_in = rxr.open_rasterio(src, masked=True, chunks=True)
        da_in = da_in.rio.write_crs(4326)

        # Reproject grid to match raster CRS if needed
        grid = grid.to_crs(da_in.rio.crs)

        # Initialize a column for pixel values
        grid["mean"] = np.nan

        # Extract the value of the raster for each grid cell
        for index, row in grid.iterrows():
            minx, miny, maxx, maxy = row["bbox"]

            # Select the raster data within the bounding box
            da_box = da_in.sel(x=slice(minx, maxx), y=slice(miny, maxy))

            # Assuming there's exactly one pixel per grid cell (there is), get its value
            pixel_value = da_box.values[0, 0, 0]  # Adjust indices if needed
            grid.at[index, "mean"] = pixel_value

        grid["date"] = date_list[i].strftime("%Y-%m-%d")
        # change values by dividing by 10 to mm/hr
        grid["mean"] /= 10
        file_df = pd.concat(
            [file_df, grid[["id", "iso3", "mean", "date"]]], axis=0
        )
        i += 1
    # Date as column, rainfall values as rows with index id, iso3
    day_wide = pd.pivot(
        file_df,
        index=["id", "iso3"],
        columns=["date"],
        values=["mean"],
    )
    # This for the headers 'Mean' and 'id'
    day_wide.columns = day_wide.columns.droplevel(0)
    day_wide.reset_index(inplace=True)
    # Max accumulated rainfall in the period selected
    day_wide["rainfall_max_24h"] = day_wide.iloc[:, 2:].max(axis=1)

    day_wide["sid"] = sid
    return day_wide[["id", "iso3", "sid", "rainfall_max_24h"]]


# Iterate process
def iterate_rainfall_dataset(iso3_list, metadata_global, grid_global):
    df_rainfall_total = pd.DataFrame()
    for iso3 in iso3_list:
        print(iso3)
        metadata_country = metadata_global[metadata_global.iso3 == iso3]
        for sid in metadata_country.sid.unique():
            print(sid)
            df_meta = metadata_country[metadata_country.sid == sid]
            # Get rainfall data at grid level
            df_rainfall = create_rainfall_dataset(
                grid_global=grid_global, df_meta=df_meta, iso3=iso3, sid=sid
            )
            df_rainfall_total = pd.concat([df_rainfall_total, df_rainfall])


if __name__ == "__main__":
    iso3_list = constant.iso3_list
    metadata_global = blob.load_csv("global_model/PPS/metadata/meta_data.csv")
    grid_global = blob.load_gpkg(
        "global_model/GRID/global_0.1_degree_grid_centroids_land_overlap.gpkg"
    )
    iterate_rainfall_dataset(
        iso3_list=iso3_list,
        metadata_global=metadata_global,
        grid_global=grid_global,
    )
