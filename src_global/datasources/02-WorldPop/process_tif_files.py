#!/usr/bin/env python3
import io
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask

from src_global.utils import blob, constant


# Function to calculate the sum of population for a given geometry
def calculate_population(geometry, raster, transform):
    # Mask the raster with the geometry
    out_image, out_transform = mask(raster, [geometry], crop=True)
    out_image = out_image[0]  # Get the single band

    # Calculate the sum of the population within the geometry
    population_sum = out_image[
        out_image > 0
    ].sum()  # Sum only positive values (valid population counts)
    return population_sum


# Load and create population dataset for each country at grid level
def pop_to_grid(grid_global, country):
    # Grid information
    grid_country = grid_global[grid_global.iso3 == country]

    # List and load the TIFF file from blob storage
    input_blob_path = (
        f"global_model/WORLDPOP/{country.lower()}_ppp_2020_UNadj.tif"
    )
    pop_raster = blob.load_tif_from_blob(input_blob_path)

    # Reproject adm2 geometries to match the raster CRS
    grid_country = grid_country.to_crs(pop_raster.crs)

    # Calculate population for each geometry in adm2
    grid_country["population"] = grid_country["geometry"].apply(
        calculate_population, args=(pop_raster, pop_raster.transform)
    )

    # Dataset
    pop_data_country = grid_country[["id", "iso3", "population"]]
    return pop_data_country


# Iterate for every country
def iterate_pop_to_grid(iso3_list, grid_global):
    df_pop_global = pd.DataFrame()
    for iso3 in iso3_list:
        # Get data
        pop_country = pop_to_grid(grid_global=grid_global, country=iso3)
        # Append data
        df_pop_global = pd.concat([df_pop_global, pop_country])

    df_pop_global = df_pop_global.reset_index(drop=True)

    # Save to blob (in chunks)
    chunk_size = 100000  # Adjust chunk size as necessary
    blob_name_template = "pop_grid_global_100m"

    # Assuming impact_data is your DataFrame
    blob.upload_in_chunks(
        dataframe=df_pop_global,
        chunk_size=chunk_size,
        blob=blob,
        folder="WORLDPOP/processed_pop",
        blob_name_template=blob_name_template,
    )


if __name__ == "__main__":
    # Define countries to download pop data
    iso3_list = constant.iso3_list

    # Load grid
    blob_dir = "global_model/GRID/global_0.1_degree_grid_land_overlap.gpkg"
    grid_global = blob.load_gpkg(blob_dir)

    # Run main function
    iterate_pop_to_grid(iso3_list=iso3_list, grid_global=grid_global)
