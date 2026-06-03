import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove personal and metadata columns that are 
    not used for modelling.
    """
    df = df.copy()

    df = df.drop(
        columns=["name_1", "name_2", "street"],
        errors="ignore"
    )

    return df


def clean_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean categorical variables and ensure that postal codes 
    have a string type.
    """
    df = df.copy()

    if "merch" in df.columns:
        df["merch"] = (
            df["merch"]
            .astype(str)
            .str.replace("fraud_", "", regex=False)
        )

    if "post_code" in df.columns:
        df["post_code"] = df["post_code"].astype(str)

    return df


def add_time_features(
    df: pd.DataFrame,
    drop_transaction_time: bool = True,
) -> pd.DataFrame:
    """
    Extract time-based features from transaction timestamp.
    """
    df = df.copy()

    # convert transaction_time var to datetime
    df["transaction_time"] = pd.to_datetime(
        df["transaction_time"]
    )
    dt = df["transaction_time"].dt

    # extract time components
    df["hour"] = dt.hour
    df["year"] = dt.year
    df["month"] = dt.month
    df["day_of_month"] = dt.day
    df["day_of_week"] = dt.dayofweek

    # is night/weekend
    df["is_night"] = df["hour"].isin(
        [0, 1, 2, 3, 4, 5]
    ).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(
        [5, 6]
    ).astype(int)

    # encode cyclic time features via sin/cos
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # categorical period of day
    df["day_period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)

    if drop_transaction_time:
        df = df.drop(columns=["transaction_time"], errors="ignore")

    return df


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create geographic distance features between customer 
    and merchant.
    """
    df = df.copy()

    # absolute coordinate diff
    df["lat_diff"] = (df["lat"] - df["merchant_lat"]).abs()
    df["lon_diff"] = (df["lon"] - df["merchant_lon"]).abs()

    # vectorized haversine distance in km
    earth_radius = 6371.0

    # convert degrees to radians
    lat1 = np.radians(df["lat"].astype(float))
    lon1 = np.radians(df["lon"].astype(float))
    lat2 = np.radians(df["merchant_lat"].astype(float))
    lon2 = np.radians(df["merchant_lon"].astype(float))

    # coordinate diff in radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = earth_radius * c

    # drop raw coordinates after creating distance features
    df = df.drop(
        columns=["lat", "lon", "merchant_lat", "merchant_lon"],
        errors="ignore",
    )

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create amount, population, and distance transformations.
    """
    df = df.copy()

    # log transforms
    df["amount_log"] = np.log1p(df["amount"])
    df["population_city_log"] = np.log1p(df["population_city"])
    df["distance_log"] = np.log1p(df["distance"])

    # ratio features: transaction size relative to 
    # distance and city population
    df["amount_per_km"] = df["amount"] / (df["distance"] + 1)
    df["amount_per_population"] = (
        df["amount"] / (df["population_city"] + 1)
    )

    # bucketized amount as cat feature
    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[-1, 10, 25, 50, 100, 250, 500, np.inf],
        labels=[
            "0_10",
            "10_25",
            "25_50",
            "50_100",
            "100_250",
            "250_500",
            "500_plus"
        ],
    ).astype(str)

    # bucketized dist as cat feature
    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[-1, 1, 5, 25, 50, 75, 100, np.inf],
        labels=[
            "0_1",
            "1_5",
            "5_25",
            "25_50",
            "50_75",
            "75_100",
            "100_plus"
        ],
    ).astype(str)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical interaction features from transaction 
    context.
    """
    df = df.copy()

    # category-time interaction
    df["cat_id_hour"] = (
        df["cat_id"].astype(str)
        + "_"
        + df["hour"].astype(str)
    )

    # category-period interaction
    df["cat_id_period"] = (
        df["cat_id"].astype(str)
        + "_"
        + df["day_period"].astype(str)
    )

    # category-weekend interaction
    df["cat_id_weekend"] = (
        df["cat_id"].astype(str)
        + "_"
        + df["is_weekend"].astype(str)
    )

    # state-category interaction
    df["state_cat_id"] = (
        df["us_state"].astype(str)
        + "_"
        + df["cat_id"].astype(str)
    )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature-engineering pipeline to
    extract/create features
    """
    df = df.copy()

    df = drop_unused_columns(df)
    df = clean_categorical_features(df)
    df = add_time_features(df)
    df = add_distance_features(df)
    df = add_amount_features(df)
    df = add_interaction_features(df)

    return df
