# CardGuard - Feature Engineering

import numpy as np
import pandas as pd


# Haversine Distance

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographic coordinates.

    Returns distance in kilometers.
    """

    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = ( np.sin(dlat / 2) ** 2+ np.cos(lat1)* np.cos(lat2)* np.sin(dlon / 2) ** 2)
    return R * (2 * np.arcsin(np.sqrt(a)))

# Main Feature Engineering Function

def create_features(df):
    """
    Apply CardGuard feature engineering to raw transaction data.
    IMPORTANT:
    The dataframe must contain transactions in chronological
    order. Historical features are calculated using previous
    transactions only.
    """
    df = df.copy()
    # 1. Convert Date Columns

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"],errors="coerce")

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime( df["dob"],errors="coerce")

    # 2. Sort Chronologically

    if "trans_date_trans_time" in df.columns:
        df = (df.sort_values("trans_date_trans_time").reset_index(drop=True))

    # TIME FEATURES

    if "trans_date_trans_time" in df.columns:
        # 3. Calendar Features

        df["transaction_year"] = (df["trans_date_trans_time"].dt.year)
        df["transaction_month"] = (df["trans_date_trans_time"].dt.month)
        df["transaction_day"] = (df["trans_date_trans_time"].dt.day)
        df["transaction_hour"] = (df["trans_date_trans_time"].dt.hour)
        df["transaction_dayofweek"] = (df["trans_date_trans_time"].dt.dayofweek)

        # 4. Weekend

        df["is_weekend"] = (df["transaction_dayofweek"] >= 5).astype(int)

        # 5. Night

        df["is_night"] = (df["transaction_hour"] < 6).astype(int)

        # 6. Business Hour

        df["is_business_hour"] = ((df["transaction_hour"] >= 9)&(df["transaction_hour"] < 18)).astype(int)

        # 7. Cyclical Hour Features

        df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)

        # 8. Cyclical Day-of-Week Features

        df["dayofweek_sin"] = np.sin(2 * np.pi * df["transaction_dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["transaction_dayofweek"] / 7)

    # AGE

    if ("trans_date_trans_time" in df.columns and "dob" in df.columns):
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

    # AMOUNT FEATURES

    if "amt" in df.columns:
        # Log Amount

        df["amt_log"] = np.log1p(df["amt"])
        # Amount Bucket

        bins = [-np.inf,
            10,
            50,
            100,
            250,
            500,
            1000,
            np.inf]

        labels = ["very_low","low","medium","high","very_high","extreme","very_extreme"]

        df["amount_bucket"] = pd.cut(df["amt"],bins=bins,labels=labels,include_lowest=True)

    # CUSTOMER → MERCHANT DISTANCE

    required_geo_columns = ["lat","long","merch_lat","merch_long"]

    if all(col in df.columns
        for col in required_geo_columns):

        df["customer_merchant_distance_km"] = (haversine_distance(
                df["lat"],
                df["long"],
                df["merch_lat"],
                df["merch_long"]))

    # PREVIOUS TRANSACTION FEATURES

    if "cc_num" in df.columns:
        # Previous Transaction Amount
        if "amt" in df.columns:
            df["previous_transaction_amt"] = (df.groupby("cc_num")["amt"].shift(1))


        # Previous Transaction Time

        if "trans_date_trans_time" in df.columns:
            df["previous_transaction_time"] = (df.groupby("cc_num")["trans_date_trans_time"].shift(1))

            # Time Since Previous Transaction
            df["time_since_previous_transaction_minutes"] = (
                df["trans_date_trans_time"].sub(df["previous_transaction_time"]).dt.total_seconds()/ 60)

    # PREVIOUS MERCHANT LOCATION
    if "cc_num" in df.columns:
        if "merch_lat" in df.columns:
            df["previous_merch_lat"] = (df.groupby("cc_num")["merch_lat"].shift(1))
        if "merch_long" in df.columns:
            df["previous_merch_long"] = (df.groupby("cc_num")["merch_long"].shift(1))

        # Distance From Previous Merchant

        if all(col in df.columns for col in ["previous_merch_lat","previous_merch_long","merch_lat","merch_long"]):
            
            df["distance_from_previous_location_km"] = (haversine_distance(df["previous_merch_lat"],
                df["previous_merch_long"],
                df["merch_lat"],
                df["merch_long"]))

    # LOCATION VELOCITY
    if all(col in df.columns for col in [
            "distance_from_previous_location_km",
            "time_since_previous_transaction_minutes"]):

        df["location_velocity_kmph"] = np.where(df["time_since_previous_transaction_minutes"] > 0,
            df["distance_from_previous_location_km"]/(df["time_since_previous_transaction_minutes"]/ 60),np.nan)
    
    # CARD TRANSACTION NUMBER
    
    if "cc_num" in df.columns: df["card_transaction_number"] = (df.groupby("cc_num").cumcount())

    # CARD / MERCHANT HISTORY
    if ("cc_num" in df.columns and "merchant" in df.columns):
        df["card_merchant_count"] = (df.groupby(["cc_num", "merchant"]).cumcount())
        df["is_new_merchant"] = (df["card_merchant_count"] == 0).astype(int)

    # CARD / CITY HISTORY

    if ("cc_num" in df.columns and "city" in df.columns):
        df["card_city_count"] = (df.groupby(["cc_num", "city"]).cumcount())
        df["is_new_city"] = (df["card_city_count"] == 0).astype(int)

    # CARD / CATEGORY HISTORY

    if ("cc_num" in df.columns and "category" in df.columns):
        df["card_category_count"] = (df.groupby(["cc_num", "category"]).cumcount())
        df["is_new_category"] = (df["card_category_count"] == 0).astype(int)

    # HISTORICAL AMOUNT STATISTICS

    if ("cc_num" in df.columns and "amt" in df.columns):

        # Previous Average Amount
        df["previous_avg_amount"] = (df.groupby("cc_num")["amt"].transform(lambda x:x.shift(1) .expanding().mean()))
        # Previous Standard Deviation
        df["previous_std_amount"] = (df.groupby("cc_num")["amt"].transform(lambda x:x.shift(1).expanding().std()))

        # Amount vs Historical Average

        df["amount_vs_historical_avg"] = (df["amt"]/df["previous_avg_amount"].replace(0, np.nan))
        # Amount Z-Score

        df["amount_zscore"] = ((df["amt"] - df["previous_avg_amount"])/df["previous_std_amount"].replace(0, np.nan))

    # REPLACE INFINITE VALUES

    df = df.replace([np.inf, -np.inf],np.nan)

    # REMOVE TEMPORARY COLUMNS

    temporary_columns = ["previous_transaction_time"]
    df = df.drop(columns=temporary_columns,errors="ignore")

    # REMOVE IDENTIFIER / PII COLUMNS

    identifier_columns = [
        "cc_num",
        "trans_num",
        "Unnamed: 0",
        "first",
        "last",
        "street"]
    df = df.drop(columns=identifier_columns,errors="ignore")

    # RETURN FEATURE-ENGINEERED DATA
    return df