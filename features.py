import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def add_features(df):
    df["increase_stock"] = df["increase_stock"].map({
        "high_bike_demand": 1,
        "low_bike_demand": 0
    })

    df["weather_severity"] = (
        df["precip"] * 2
        + (df["temp"] < 5).astype(int) * 1
        + (df["windspeed"] > 15).astype(int) * 2
        + (df["visibility"] < 5).astype(int) * 2
    )

    conditions = [
        df["hour_of_day"].between(8, 14),
        df["hour_of_day"].between(15, 19),
    ]
    values = [1, 2]

    df["rush_hour_severity"] = np.select(conditions, values, default=0)

    df["time_off"] = ((1 - df["weekday"]) | (df["holiday"]))

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    return df

def get_X_y(df):
    X = df[[
        "weather_severity",
        "rush_hour_severity",
        "hour_sin",
        "hour_cos",
        "time_off",
        "month_sin",
        "month_cos",
    ]]
    y = df["increase_stock"]
    return X, y
