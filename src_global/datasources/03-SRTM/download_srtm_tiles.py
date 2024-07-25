#!/usr/bin/env python3
import io
import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Polygon

from src_global.utils import blob, constant


# creates a bounding box and selects boxes within or touching it
def srtm_sel(polygons):
    # Get bounds of geometry
    extent = list(polygons.total_bounds)
    # Based on the bounds, select tiles
    wld_lon_extent = 180 + 180  # 180W to 180E
    wld_lat_extent = 60 + 60  # 60N to 60S
    wld_lon_box = wld_lon_extent / 72  # 72 is the number of columns of data
    wld_lat_box = wld_lat_extent / 24  # 24 is the number of rows of data
    wld_lon_start = list(range(-180, 180, int(wld_lon_box)))
    wld_lat_start = list(range(60, -60, int(wld_lat_box) * -1))
    country_lat_start = [
        n for n, i in enumerate(wld_lat_start) if i < extent[3]
    ][0]
    country_lat_end = [
        n for n, i in enumerate(wld_lat_start) if i > extent[1]
    ][-1:][0]
    country_lon_start = [
        n for n, i in enumerate(wld_lon_start) if i > extent[0]
    ][0]
    country_lon_end = [
        n for n, i in enumerate(wld_lon_start) if i > extent[2]
    ][0]
    # Lat and lon lists
    lat_list = list(range(country_lat_start, country_lat_end + 1))
    lon_list = list(range(country_lon_start, country_lon_end + 1))
    # There is no 00 tile, it starts again from 72
    for i in range(len(lon_list)):
        if lon_list[i] == 0:
            lon_list[i] = 72
    # Appropiate way of calling the file
    country_list = [
        "%02d" % x + "_" + "%02d" % y for x in lon_list for y in lat_list
    ]
    country_file_list = ["srtm_" + file + ".zip" for file in country_list]
    return country_file_list


# Define a function to adjust the longitude of a single polygon
# the way srtm data likes it --> (-360, 0)
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)

    # Adjust longitudes from [-180, 180) to [-360, 0)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 0:
            coords[i] = (lon - 360, lat)

    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


# Downlaod SRTM tiles for specific country
def download_tiles(
    iso3,
    global_grid,
    base_url,
    PROJECT_PREFIX="global_model",
    transform_geometry=False,
):
    # Select country geometry
    country_adm2 = global_grid[global_grid.iso3 == iso3].copy()
    if transform_geometry:
        # Apply the adjust_longitude function to each geometry in the DataFrame
        country_adm2["geometry"] = country_adm2["geometry"].apply(
            adjust_longitude
        )
    # Define tiles to download
    country_file_list = srtm_sel(polygons=country_adm2)
    # Downlaod
    for file in country_file_list:
        try:
            req = requests.get(base_url + file, verify=True, stream=True)
            with zipfile.ZipFile(io.BytesIO(req.content)) as zObj:
                fileNames = zObj.namelist()
                for fileName in fileNames:
                    if fileName.endswith("tif"):
                        content = zObj.open(fileName).read()
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".tif"
                        ) as temp_file:
                            temp_file.write(content)
                            temp_file_path = temp_file.name
                            blob_name = f"{PROJECT_PREFIX}/SRTM/{iso3}/{fileName}"
                            # Check if file is already on blob
                            check = len(
                                blob.list_container_blobs(
                                    name_starts_with=blob_name
                                )
                            )
                            if check == 1:
                                print(
                                    f"{fileName} tile already in the {iso3} folder"
                                )
                                pass
                            else:
                                # print(f"Uploading {fileName} to {iso3} folder")
                                blob.upload_tif_to_blob(
                                    file_path=temp_file_path, blob_name=blob_name
                                )
        except:
            #Ocean tile
            pass


# Iterate downloading process
def download_process(global_grid):
    # Define url and list of countries
    iso3_list = constant.iso3_list
    base_url = constant.srtm_url
    # Iterate
    i = 0
    for iso3 in iso3_list:
        print(f"Getting data for {iso3}, {i+1}/{len(iso3_list)}")
        i += 1
        if (iso3 == "FJI") or (iso3 == "RUS"):
            transform_geometry = True
        else:
            transform_geometry = False

        download_tiles(
            iso3=iso3,
            global_grid=global_grid,
            base_url=base_url,
            transform_geometry=transform_geometry,
        )


if __name__ == "__main__":
    # Load global grid dataset
    global_grid = blob.load_gpkg(
        "global_model/GRID/global_0.1_degree_grid_centroids_land_overlap.gpkg"
    )
    # Download data
    download_process(global_grid)
