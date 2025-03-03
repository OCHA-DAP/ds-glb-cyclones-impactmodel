#!/usr/bin/env python3
import numpy as np
import pandas as pd

# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBClassifier, XGBRegressor

from src_global.utils import blob

# Features
features = [
    "wind_speed",
    "track_distance",
    # "total_buildings",
    "population",
    "rainfall_max_24h",
    "flood_risk",
    "landslide_risk_sum",
    "number_of_event",
    "events_in_last_5_years",
    # "coast_length",
    # "with_coast",
    # "mean_elev",
    # "mean_slope",
    # "mean_rug",
    # "IWI",
]

features_all = features + [
    "perc_affected_pop_grid_region",
    "DisNo.",
    "sid",
    "id",
    "iso3",
    "typhoon",
]
# df_global = pd.read_csv('/home/fmoss/GLOBAL MODEL/ds-glb-cyclones-impactmodel/src_global/datasources/model_input_data.csv')
# df_global = pd.read_csv('/home/fmoss/GLOBAL MODEL/ds-glb-cyclones-impactmodel/src_global/datasources/model_input_data_weather_constraints.csv')
# df_global['typhoon'] = df_global['typhoon_name'] + df_global['typhoon_year'].astype(str)

# def load_and_process_data(cell_size):
#     # Load data
#     df_dynamic = blob.load_csv(f'ds-aa-hti-hurricanes/GRID_CELL_SIZE/{cell_size}/dynamic_features.csv')
#     df_stationary = blob.load_csv(f'ds-aa-hti-hurricanes/GRID_CELL_SIZE/{cell_size}/stationary_features.csv')

#     # Merge dataframes
#     df_hti = df_stationary.merge(df_dynamic)

#     # ADM1 region by id
#     mun_id = df_hti[['grid_point_id', 'ADM1_PCODE']].drop_duplicates().reset_index(drop=True)

#     # Return both processed dataframes with dynamically generated names
#     return mun_id, df_hti

# mun_id, df_hti = load_and_process_data(0.1)

# # Rename some features
# df_hti = df_hti.rename(
#     {'total_pop':'population',
#      'track_id':'sid',
#      'grid_point_id':'id',},
#     axis=1)
# # Add iso3 information
# df_hti['iso3'] = 'HTI'

# df_global_fixed = pd.concat([df_hti, df_global])
# df_global_fixed_red = df_global_fixed[features_all]
# df_global_fixed_red = df_global[features_all].copy()

df_global = pd.read_csv(
    "/home/fmoss/GLOBAL MODEL/ds-glb-cyclones-impactmodel/src_global/datasources/model_input_data.csv"
)
df_global = df_global.drop_duplicates(
    subset=["sid", "iso3", "id", "perc_affected_pop_grid_region"]
)

# Drop particular events
df_global["typhoon"] = df_global.typhoon_name + df_global.typhoon_year.astype(
    "str"
)
weird_events = pd.read_csv(
    "/home/fmoss/GLOBAL MODEL/ds-glb-cyclones-impactmodel/local_files/weird_cases.csv"
)
weird_events["iso3"] = weird_events["GID_0"]

# Filter out the events
df_global_filtered = df_global[
    ~df_global.typhoon.isin(weird_events.typhoon)
].reset_index(drop=True)


def reg_model_LOOCV(df, features, following="HTI"):
    """
    Linear regression model using Leave-One-Out Cross-Validation (LOOCV).

    Parameters:
    - df: The dataframe containing the data.
    - features: The list of feature column names for training.
    - following: The ISO3 code of the country for testing (default is 'HTI').

    Returns:
    - y_test_typhoon: List of actual values of the test set.
    - y_pred_typhoon: List of predicted values from the linear regressor.
    """

    # Dataframe for testing: Default is Haiti (HTI)
    aux = df[["iso3", "sid"]].drop_duplicates()
    aux = aux[aux.iso3 == following]

    y_test_typhoon = []
    y_pred_typhoon = []

    for sid, iso3 in zip(aux["sid"], aux["iso3"]):
        """PART 1: Train/Test Split"""

        # LOOCV - Test set for the specific event
        df_test = df[(df["sid"] == sid) & (df["iso3"] == iso3)]

        # Training set excluding the specific event
        df_train = df[(df["sid"] != sid) & (df["iso3"] != iso3)]

        # Split X and y from dataframe features
        X_train = df_train[features]
        X_test = df_test[features]

        y_train = df_train["perc_affected_pop_grid_grid"]
        y_test = df_test["perc_affected_pop_grid_grid"]

        """ PART 2: Linear Regressor """

        # Create a Linear Regressor model
        reg = LinearRegression()

        # Fit the model on the training data
        reg.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = reg.predict(X_test)

        # Save y_test and y_pred for later analysis
        y_test_typhoon.append(y_test)
        y_pred_typhoon.append(y_pred)

    return y_test_typhoon, y_pred_typhoon


def xgb_model_LOOCV(
    df,
    features,
    following="HTI",
    weight=2,
    target_name="perc_affected_pop_grid_grid",
    validation_events=None,
):
    # Dataframe for testing: HAITI (default)
    aux = df[["iso3", "sid", "DisNo."]].drop_duplicates()
    aux = aux[aux["DisNo."].isin(validation_events)]
    aux = aux[aux.iso3 == following]

    y_test_typhoon = []
    y_pred_typhoon = []

    # Custom objective that penalizes more over-predictions
    def asymmetric_loss(y_true, y_pred):
        residual = y_pred - y_true
        # Penalize over-predictions more heavily
        grad = np.where(residual > 0, 2 * residual, residual)
        hess = np.where(residual > 0, 2.0, 1.0)  # Second derivative
        return grad, hess

    def custom_track_loss(y_true, y_pred, track_distance):
        # Ensure that y_true and y_pred are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate weights based on track_distance (e.g., exponential weighting)
        gamma = 0.1
        weights = np.exp(
            gamma * np.array(track_distance)
        )  # Convert track_distance to numpy array if it's a Series

        # Calculate residuals
        residual = y_pred - y_true

        # Weighted gradients and Hessians
        grad = weights * residual
        hess = weights  # For squared error, Hessian is just the weight

        return grad, hess

    for sid, iso3, DisNo in zip(aux["sid"], aux["iso3"], aux["DisNo."]):
        """PART 1: Train/Test"""

        # LOOCV
        df_test = df[
            (df["sid"] == sid) & (df["iso3"] == iso3) & (df["DisNo."] == DisNo)
        ]  # Test set: HTI event
        df_train = df[
            (df["sid"] != sid) & (df["iso3"] != iso3) & (df["DisNo."] != DisNo)
        ]  # Train set: everything else

        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train[target_name]
        y_test = df_test[target_name]

        # Class weight
        weights = np.where(df_train["iso3"] == following, weight, 1)

        """ PART 2: XGB regressor """
        # create an XGBoost Regressor
        xgb = XGBRegressor(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=0.8,
            colsample_bynode=0.8,
            colsample_bytree=0.8,
            gamma=3,
            eta=0.01,
            importance_type="gain",
            learning_rate=0.1,
            max_delta_step=0,
            max_depth=4,
            min_child_weight=1,
            missing=1,
            n_estimators=100,
            early_stopping_rounds=10,
            n_jobs=1,
            nthread=None,
            # objective="reg:squarederror",
            # objective=asymmetric_loss,
            objective=lambda y_true, y_pred: custom_track_loss(
                y_true, y_pred, X_train["track_distance"].values
            ),
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=None,
            silent=None,
            subsample=0.8,
            verbosity=0,
            # eval_metric=["rmse", "logloss"],
            eval_metric="rmse",
            random_state=0,
        )

        # Fit the model
        eval_set = [(X_train, y_train)]
        # xgb.fit(X_train, y_train,
        #         eval_set=eval_set, verbose=False,
        #         sample_weight=weights)
        xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # make predictions on Fiji
        y_pred = xgb.predict(X_test)

        # Save y_test y_pred
        y_test_typhoon.append(y_test)
        y_pred_typhoon.append(y_pred)

    return y_test_typhoon, y_pred_typhoon


def xgb_model_LOOCV_boot(
    df,
    features,
    following="HTI",
    weight=2,
    cycle=10,
    frac=0.8,
    shuffle=True,
    bootstrapping=True,
    target_name="perc_affected_pop_grid_grid",
):
    # Dataframe for testing: HAITI (default)
    aux = df[["iso3", "sid", "DisNo."]].drop_duplicates()
    aux = aux[aux.iso3 == following]

    y_test_typhoon = []
    y_pred_typhoon = []
    for i in range(cycle):
        for sid, iso3, DisNo in zip(aux["sid"], aux["iso3"], aux["DisNo."]):
            """PART 1: Train/Test"""

            # LOOCV
            df_test = df[
                (df["sid"] == sid)
                & (df["iso3"] == iso3)
                & (df["DisNo."] == DisNo)
            ]  # Test set: HTI event
            df_train = df[
                (df["sid"] != sid)
                & (df["iso3"] != iso3)
                & (df["DisNo."] != DisNo)
            ]  # Train set: everything else

            if shuffle:
                df_train = df_train.sample(frac=1)
            if bootstrapping:
                df_train = df_train.sample(frac=frac)

            # Split X and y from dataframe features
            X_test = df_test[features]
            X_train = df_train[features]

            y_train = df_train[target_name]
            y_test = df_test[target_name]

            # Class weight
            weights = np.where(df_train["iso3"] == following, weight, 1)

            """ PART 2: XGB regressor """
            # create an XGBoost Regressor
            xgb = XGBRegressor(
                base_score=0.5,
                booster="gbtree",
                colsample_bylevel=0.8,
                colsample_bynode=0.8,
                colsample_bytree=0.8,
                gamma=3,
                eta=0.01,
                importance_type="gain",
                learning_rate=0.1,
                max_delta_step=0,
                max_depth=4,
                min_child_weight=1,
                missing=1,
                n_estimators=100,
                early_stopping_rounds=10,
                n_jobs=1,
                nthread=None,
                objective="reg:squarederror",
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                seed=None,
                silent=None,
                subsample=0.8,
                verbosity=0,
                eval_metric=["rmse", "logloss"],
                random_state=0,
            )

            # Fit the model
            eval_set = [(X_train, y_train)]
            xgb.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                sample_weight=weights,
            )

            # make predictions on Fiji
            y_pred = xgb.predict(X_test)

            # Save y_test y_pred
            y_test_typhoon.append(y_test)
            y_pred_typhoon.append(y_pred)

    return y_test_typhoon, y_pred_typhoon


if __name__ == "__main__":
    country_of_interest = "PHL"
    boot = False
    validation_events = [
        "2018-0341-PHL",
        # '2006-0517-PHL',
        # '2014-0479-PHL',
        # '2009-0422-PHL',
        # '2014-0227-PHL',
        "2008-0249-PHL",
        # '2009-0414-PHL',
        "2012-0500-PHL",
        "2021-0813-PHL",
        "2013-0433-PHL",
        "2005-0120-PHL",
        "2021-0595-PHL",
        "2020-0452-PHL",
        "2017-0422-PHL",
        "2012-0463-PHL",
        "2008-0540-PHL",
        "2015-0176-PHL",
        "2019-0549-PHL",
        "2014-0240-PHL",
        "2004-0218-PHL",
    ]

    # y_test, y_pred = reg_model_LOOCV(
    #     df_global_fixed_red,
    #     features,
    #     following='HTI')
    if boot:
        y_test, y_pred = xgb_model_LOOCV_boot(
            df=df_global,
            features=features,
            following=country_of_interest,
            weight=2,
            target_name="perc_affected_pop_grid_region",
        )

        # Save them to a .npz file (which stores multiple arrays)
        np.savez(
            f"{country_of_interest}_xgb_weather_constraints_boot.npz",
            array1=y_test,
            array2=y_pred,
        )
    else:
        y_test, y_pred = xgb_model_LOOCV(
            df=df_global,
            features=features,
            following=country_of_interest,
            weight=2,
            target_name="perc_affected_pop_grid_region",
            validation_events=validation_events,
        )

        # Save them to a .npz file (which stores multiple arrays)
        np.savez(
            f"{country_of_interest}_xgb_no_boot_subset_validation_custom_objective_track_distance.npz",
            array1=y_test,
            array2=y_pred,
        )
