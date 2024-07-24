#!/usr/bin/env python3
import io
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from shapely.geometry import Polygon

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
def pop_to_grid(pop_raster, grid_global, country, transform_geometry=False):
    # Grid information
    grid_country = grid_global[grid_global.iso3 == country]

    # Reproject adm2 geometries to match the raster CRS
    grid_country = grid_country.to_crs(pop_raster.crs)

    if transform_geometry:
        # Apply the adjust_longitude function to each geometry in the DataFrame
        grid_country["geometry"] = grid_country["geometry"].apply(adjust_longitude)

    # Calculate population for each geometry in adm2
    grid_country["population"] = grid_country["geometry"].apply(
        calculate_population, args=(pop_raster, pop_raster.transform)
    )

    # Dataset
    pop_data_country = grid_country[["id", "iso3", "population"]]
    return pop_data_country

# Define a function to adjust the longitude of a single polygon
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)
    
    # Adjust longitudes from [0, 360) to [-180, 180)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    
    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


# Iterate for every country
def iterate_pop_to_grid(iso3_list, grid_global):
    # Load the global TIF file from the blob storage
    input_blob_path = 'global_model/WORLDPOP/ppp_2020_1km_Aggregated.tif'
    print('Loading global tif file')
    pop_raster = blob.load_tif_from_blob(input_blob_path)
    print('Global tif file loaded')
    # Iterate
    df_pop_global = pd.DataFrame()
    i=0
    for iso3 in iso3_list:
        print(f'Getting pop data from {iso3}, {i}/{len(iso3_list)}')
        i+=1
        if (iso3 == 'FJI') or (iso3 == 'RUS'):
            # Get transformed data
            transform_geometry = True
        else:
            transform_geometry = False
        # Get data
        pop_country = pop_to_grid(pop_raster=pop_raster, 
                                    grid_global=grid_global, 
                                    country=iso3,
                                    transform_geometry=transform_geometry)
        # Append data
        df_pop_global = pd.concat([df_pop_global, pop_country])

    df_pop_global = df_pop_global.reset_index(drop=True)

    # Save to blob (in chunks)
    chunk_size = 100000  # Adjust chunk size as necessary
    blob_name_template = "pop_grid_global"

    # Assuming impact_data is your DataFrame
    print('Uploading data to blob')
    blob.upload_in_chunks(
        dataframe=df_pop_global, 
        chunk_size=chunk_size, 
        blob=blob, 
        folder = "WORLDPOP/processed_pop",
        blob_name_template=blob_name_template)


if __name__ == "__main__":
    # Define countries to download pop data
    iso3_list = constant.iso3_list

    # Load grid
    print('Loading grid cells')
    blob_dir = "global_model/GRID/global_0.1_degree_grid_land_overlap.gpkg"
    grid_global = blob.load_gpkg(blob_dir)
    print('Grid cells loaded')

    # Run main function
    iterate_pop_to_grid(iso3_list=iso3_list, grid_global=grid_global)
