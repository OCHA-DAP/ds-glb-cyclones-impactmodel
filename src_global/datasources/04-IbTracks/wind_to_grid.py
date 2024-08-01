#!/usr/bin/env python3
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString
from wind_functions import (
    add_interpolation_points,
    adjust_tracks,
    calculate_mean_for_neighbors,
    get_closest_point_index,
    windfield_to_grid,
)

from src_global.utils import blob, constant

PROJECT_PREFIX = constant.PROJECT_PREFIX

"""     Load grid data and shapefile        """
def load_data():
    # Load grid-land overlap data
    gdf = blob.load_gpkg(f"{PROJECT_PREFIX}/GRID/global_0.1_degree_grid_centroids_land_overlap.gpkg")
    # Load all grid data (include oceans)
    gdf_all = blob.load_gpkg(f"{PROJECT_PREFIX}/GRID/global_0.1_degree_grid_centroids.gpkg")

    # Load shapefile
    shp = blob.load_gpkg(f"{PROJECT_PREFIX}/SHP/global_shapefile_GID_adm2.gpkg")
    return gdf, gdf_all, shp


"""     Load impact data       """
def load_impact_data():
    impact_data = blob.get_impact_data()
    all_events = impact_data[
        ['GID_0', 'DisNo.', 'Start Year', 'Event Name', 'sid']
        ].drop_duplicates()
    return all_events


"""     Download and proccess tracks      """
def get_storm_tracks(all_events):
    sel_ibtracs = []
    problematic_sid = []
    for sid in all_events.sid:
        try:
            sel_ibtracs.append(TCTracks.from_ibtracs_netcdf(storm_id=sid))
        except:
            problematic_sid.append(sid)
            pass

    # Interpolation proccess with current data (Doesnt make a difference if the data has few datapoints)
    # obs: .interp(x0,x,f(x)) gives the position of x0 in the fitting of (x,f(x))
    # obs: daterange consider the track between certain intervals as discrete points instead of a continuous
    tc_tracks = TCTracks()
    for track in sel_ibtracs:
        tc_track = track.get_track()
        tc_track.interp(
            time=pd.date_range(
                tc_track.time.values[0], tc_track.time.values[-1], freq="30T"
            )
        )
        tc_tracks.append(tc_track)
    return tc_tracks, problematic_sid


# Add interpolation points (smooth the track)
def proccess_storm_tracks(tc_tracks):
    tracks = TCTracks()
    for i in range(len(tc_tracks.get_track())):
        # Define relevant features
        track_xarray = tc_tracks.get_track()[i]
        time_array = np.array(track_xarray.time)
        time_step_array = np.array(track_xarray.time_step)
        lat_array = np.array(track_xarray.lat)
        lon_array = np.array(track_xarray.lon)
        max_sustained_wind_array = np.array(track_xarray.max_sustained_wind)
        central_pressure_array = np.array(track_xarray.central_pressure)
        environmental_pressure_array = np.array(
            track_xarray.environmental_pressure
        )
        r_max_wind_array = np.array(track_xarray.radius_max_wind)
        r_oci_array = np.array(track_xarray.radius_oci)

        # Define new variables
        # Interpolate every important data
        w = max_sustained_wind_array.copy()
        t = time_array.copy()
        t_step = time_step_array.copy()
        lat = lat_array.copy()
        lon = lon_array.copy()
        cp = central_pressure_array.copy()
        ep = environmental_pressure_array.copy()
        rmax = r_max_wind_array.copy()
        roci = r_oci_array.copy()

        # Define the number of points to add between each pair of data points
        num_points_between = 2

        # Add interpolation points to regulat variables
        new_w = add_interpolation_points(w, num_points_between)
        new_t_step = add_interpolation_points(t_step, num_points_between)
        new_lat = add_interpolation_points(lat, num_points_between)
        new_lon = add_interpolation_points(lon, num_points_between)
        new_cp = add_interpolation_points(cp, num_points_between)
        new_ep = add_interpolation_points(ep, num_points_between)
        new_rmax = add_interpolation_points(rmax, num_points_between)
        new_roci = add_interpolation_points(roci, num_points_between)

        # Add interpolation points to time variables
        timestamps = np.array(
            [date.astype("datetime64[s]").astype("int64") for date in t]
        )  # Convert to seconds
        new_t = add_interpolation_points(timestamps, num_points_between)
        new_t = [
            np.datetime64(int(ts), "s") for ts in new_t
        ]  # Back to datetime format

        # Define dataframe
        df_t = pd.DataFrame(
            {
                "MeanWind": new_w,
                "Pressure_env": new_ep,
                "Pressure": new_cp,
                "Latitude": new_lat,
                "Longitude": new_lon,
                "RadiusMaxWinds": new_rmax,
                "RadiusOCI": new_roci,
                "time_step": new_t_step,
                "basin": np.array(
                    [np.array(track_xarray.basin)[0]] * len(new_t)
                ),
                "forecast_time": new_t,
                "Category": track_xarray.category,
            }
        )

        # Define a custom id
        custom_idno = track_xarray.id_no
        custom_sid = track_xarray.sid
        name = track_xarray.name  # + ' interpolated'

        # Define track as climada likes it
        track = TCTracks()
        track.data = [
            adjust_tracks(
                df_t, name=name, custom_sid=custom_sid, custom_idno=custom_idno
            )
        ]

        # Tracks modified
        tracks.append(track.get_track())
    return tracks


# Convert DataFrame to CSV in-memory
def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

# Features for the model at grid level
def create_windfield_features(tracks, cent_all, gdf_all, gdf, iso3):
    # TropCyclone class
    tc_all = TropCyclone.from_tracks(
        tracks, centroids=cent_all, store_windfields=True, intensity_thres=0
    )

    # Create grid-level windfield
    df_windfield_interpolated = windfield_to_grid(
        tc=tc_all, tracks=tracks, grids=gdf_all
    )

    # Overlap
    df_windfield_interpolated_overlap = df_windfield_interpolated[
        df_windfield_interpolated.grid_point_id.isin(gdf.id)
    ]

    return df_windfield_interpolated_overlap


"""     Metadata of the events      """
def create_metadata(tracks, all_events, shp, iso3):
    df_metadata_fixed = pd.DataFrame()
    for i in range(len(tracks.data)):
        # Basics
        startdate = np.datetime64(np.array(tracks.data[i].time[0]), "D")
        enddate = np.datetime64(np.array(tracks.data[i].time[-1]), "D")
        name = tracks.data[i].name
        year = tracks.data[i].sid[:4]
        nameyear = name + year

        # For the landfall
        # Track path
        tc_track = tracks.get_track()[i]
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        track_points = gpd.GeoDataFrame(geometry=points)

        # Set crs
        track_points.crs = shp.crs

        try:
            # intersection --> Look for first intersection == landfall
            min_index = shp.sjoin(track_points)["index_right"].min()

            landfalldate = np.datetime64(
                np.array(tracks.data[i].time[min_index]), "D"
            )
            landfall_time = str(
                np.datetime64(np.array(tracks.data[i].time[min_index]), "s")
            ).split("T")[1]
        except:
            # No landfall situation --> Use closest point to shapefile
            closest_point_index = get_closest_point_index(track_points, shp)
            landfalldate = np.datetime64(
                np.array(tracks.data[i].time[closest_point_index]), "D"
            )
            landfall_time = str(
                np.datetime64(
                    np.array(tracks.data[i].time[closest_point_index]), "s"
                )
            ).split("T")[1]

        # Create df
        df_aux = pd.DataFrame(
            {
                "iso3": [iso3],
                "typhoon": [nameyear],
                "startdate": [startdate],
                "enddate": [enddate],
                "landfalldate": [landfalldate],
                "landfall_time": [landfall_time],
            }
        )
        df_metadata_fixed = pd.concat([df_metadata_fixed, df_aux])

    # Reset index
    df_metadata_fixed = df_metadata_fixed.reset_index(drop=True)
    # Merge with landfall information
    df_metadata_fixed_complete = df_metadata_fixed.merge(all_events)

    return df_metadata_fixed_complete

def process_ibtracks_data(iso3_list):
    # Load grid & shapefile data
    (gdf_global, 
     gdf_all_global, 
     shp_global) = load_data()
    # Load impact data
    all_events_global = load_impact_data()

    # Iterate for every country
    df_windspeed_complete = pd.DataFrame()
    df_meta_complete = pd.DataFrame()
    missing_storms_ibtracks = []
    i=0
    for iso3 in iso3_list:
        # Grid geometries for the country
        gdf = gdf_global[gdf_global.iso3 == iso3]
        gdf_all = gdf_all_global[gdf_all_global.iso3 == iso3]
        # Centroids
        cent = Centroids.from_geodataframe(gdf)  # grid-land overlap
        cent_all = Centroids.from_geodataframe(gdf_all)  # include oceans
        # Country geometry
        shp = shp_global[shp_global.GID_0 == iso3]
        # Events subset
        all_events = all_events_global[all_events_global.GID_0 == 'iso3']

        # Get tracks
        tc_tracks, problematic_sid = get_storm_tracks(all_events=all_events)
        # Proccess tracks
        tracks = proccess_storm_tracks(tc_tracks=tc_tracks)
        # Create features
        df_wind = create_windfield_features(
            tracks=tracks, 
            cent_all=cent_all, 
            gdf_all=gdf_all, 
            gdf=gdf,
            iso3=iso3
            )
        # Create metadata
        df_meta = create_metadata(
            tracks=tracks, 
            all_events=all_events, 
            shp=shp,
            iso3=iso3
            )
        
        # Append data
        df_windspeed_complete = pd.concat([df_windspeed_complete, df_wind])
        df_meta_complete = pd.concat([df_meta_complete, df_meta])
        missing_storms_ibtracks.append(problematic_sid)
        print(f"Dataset created {i+1}/{len(iso3_list)}")
        i+=1
    # Reset index
    df_windspeed_complete = df_windspeed_complete.reset_index(drop=True)
    df_meta_complete = df_meta_complete.reset_index(drop=True)
    # Dataframe of missing data (if there is)
    try:
        df_missing = pd.DataFrame(missing_storms_ibtracks, columns=['sid']).dropna()
        # Save to blob
        csv_data = df_missing.to_csv(index=False)
        blob.upload_blob_data(
            blob_name=PROJECT_PREFIX
            + f"windfield/missing_sid.csv",
            data=csv_data,
            )
    except:
        pass
        
    # Save csvs to blob
    csv_data_wind = df_windspeed_complete.to_csv(index=False)
    blob.upload_blob_data(
        blob_name=PROJECT_PREFIX
        + f"windfield/wind_data_complete.csv",
        data=csv_data_wind,
    )
    csv_data_meta = df_meta_complete.to_csv(index=False)
    blob.upload_blob_data(
        blob_name=PROJECT_PREFIX
        + f"windfield/meta_data_complete.csv",
        data=df_meta_complete,
    )


if __name__ == "__main__":
    # Run main function
    iso3_list = constant.iso3_list
    process_ibtracks_data(iso3_list)