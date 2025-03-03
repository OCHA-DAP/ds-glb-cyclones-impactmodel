#!/usr/bin/env python3

import numpy as np
import pandas as pd

from src_global.utils import blob, constant

def get_wind_data(PROJECT_PREFIX='global_model'):
    # Load data (is in chunks)
    filenames = [
        f"wind_data_part_{i}.csv" for i in range(1, 184)
    ]
    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/IBTRACKS/{filename}"
        df = blob.load_csv(csv_path=csv_path)
        dataframes.append(df)
    wind_data = pd.concat(dataframes, ignore_index=True)
    return wind_data

def get_rain_data(PROJECT_PREFIX='global_model'):
    # Load data (is in chunks)
    filenames = [
        f"rainfall_data_part_{i}.csv" for i in range(1, 174)
    ]
    dataframes = []
    for filename in filenames:
        csv_path = f"{PROJECT_PREFIX}/PPS/{filename}"
        df = blob.load_csv(csv_path=csv_path)
        dataframes.append(df)
    rain_data = pd.concat(dataframes, ignore_index=True)
    return rain_data


def merge_weather_impact_population(wind_data, rain_data, 
                                    impact_data, population_data,
                                    static_risks, vulnerability_meta,
                                    weather_constraints=False):
    """
    Merges wind, rain, impact, and population data into a single dataset.
    """
    if weather_constraints:
        # weather features are already in the dataset
        impact_weather_merged = impact_data.copy()

    else:
        # Step 1: Merge wind and rain data (renaming 'grid_point_id' and 'track_id')
        weather_data_merged = wind_data.rename(
            {'grid_point_id': 'id', 'track_id': 'sid'}, axis=1
        ).merge(rain_data, on=['sid', 'iso3', 'id'], how='inner')
        
        # Step 2: Merge with impact data
        impact_weather_merged = weather_data_merged.merge(impact_data, on=['sid', 'iso3'], how='inner')
        
    # Step 3: Merge with population data
    input_data_merged = impact_weather_merged.merge(population_data, on=['iso3', 'id'], how='inner')
    
    # Step 4: Drop unnecessary columns and remove duplicates
    input_data_merged = input_data_merged.drop(
        ['GID_0', 'GID_1', 'GID_2'], axis=1
    ).drop_duplicates().reset_index(drop=True)

    # Step 5: Merge with flood and landslide risk metrics
    input_data_merged = input_data_merged.merge(static_risks)
    # Step 6: Add vulnerability from metadata of events
    input_data_merged = input_data_merged.merge(vulnerability_meta, how='left')
    
    return input_data_merged

def save_input_data_to_blob(weather_constraints=False):
    # Set the blob name template based on the weather_constraints parameter
    blob_name_template = "model_input_data_weather_constraints" if weather_constraints else "model_input_data"
    
    # Save to blob
    blob.upload_in_chunks(
        dataframe=input_data_merged,
        chunk_size=100000,
        blob=blob,
        blob_name_template=blob_name_template,
        folder="model_input",
        # project_prefix='global_model'
    )

if __name__ == "__main__":

    # Prompt the user to choose True or False for weather_constraints
    user_input = input("Would you like to enable weather constraints? (True/False): ")

    # Convert the input string to a boolean value
    if user_input.lower() == 'true':
        weather_constraints = True
    elif user_input.lower() == 'false':
        weather_constraints = False
    else:
        raise ValueError("Invalid input! Please enter 'True' or 'False'.")


    # load data
    impact_data = blob.get_impact_data_at_grid_level(weather_constraints=weather_constraints)
    rain_data = get_rain_data()
    wind_data = get_wind_data()
    population_data = blob.get_population_data()
    flood_risk = pd.read_csv('/home/fmoss/GLOBAL MODEL/data/FloodRisk/global_flood_risk_index.csv')
    landslide_risk = pd.read_csv('/home/fmoss/GLOBAL MODEL/data/landslides/global_landslide_risk_index.csv')
    static_risks = flood_risk.merge(landslide_risk[['id', 'iso3', 'landslide_risk_sum']])
    vulnerability_meta = pd.read_csv('/home/fmoss/GLOBAL MODEL/data/vulnerability_from_metadata/metadata_vulnerability_feature.csv')
    
    # Merge data
    input_data_merged = merge_weather_impact_population(
        wind_data, 
        rain_data, 
        impact_data, 
        population_data,
        static_risks,
        vulnerability_meta,
        weather_constraints=weather_constraints
        )
    
    save_input_data_to_blob(weather_constraints=weather_constraints)

