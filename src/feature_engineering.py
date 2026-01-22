import pandas as pd
import numpy as np

LAGS = [1, 4, 96]  # 15 min, 1 hour, 1 day

def add_time_features(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, lags):
    df = df.sort_values(["series_id", "timestamp"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("series_id")["value"].shift(lag)
    return df

def merge_weather_data(df, weather_path):
    weather = pd.read_csv(weather_path)
    weather["timestamp"] = pd.to_datetime(weather["datetime (UTC)"], utc=True).dt.tz_localize(None)
    weather = weather.drop(columns=[
        "datetime (UTC)",
        "coordinates (lat,lon)",
        "model (name)",
        "model elevation (surface)",
        "utc_offset (hrs)"
    ])
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp",
        direction="backward"
    )
    return merged

def engineer_features(processed_dir, weather_path, output_dir):
    print("Loading processed data...")
    train = pd.read_csv(f"{processed_dir}/train_processed.csv", parse_dates=["timestamp"])
    val = pd.read_csv(f"{processed_dir}/validation_processed.csv", parse_dates=["timestamp"])
    test = pd.read_csv(f"{processed_dir}/test_processed.csv", parse_dates=["timestamp"])
    
    print("Adding time features...")
    train = add_time_features(train)
    val = add_time_features(val)
    test = add_time_features(test)
    
    print("Adding lag features...")
    train = add_lag_features(train, LAGS)
    val = add_lag_features(val, LAGS)
    test = add_lag_features(test, LAGS)
    
    print("Merging weather data...")
    train = merge_weather_data(train, weather_path)
    val = merge_weather_data(val, weather_path)
    test = merge_weather_data(test, weather_path)
    
    print("Handling missing values...")
    lag_cols = [c for c in train.columns if c.startswith("lag_")]
    train = train.dropna(subset=lag_cols)
    val = val.dropna(subset=lag_cols)
    test = test.dropna(subset=lag_cols)
    
    weather_cols = ["temperature (degC)", "wind_speed (m/s)", "surface_solar_radiation (W/m^2)"]
    train[weather_cols] = train[weather_cols].fillna(method="ffill")
    val[weather_cols] = val[weather_cols].fillna(method="ffill")
    test[weather_cols] = test[weather_cols].fillna(method="ffill")
    
    print("Saving engineered data...")
    train.to_csv(f"{output_dir}/train_features.csv", index=False)
    val.to_csv(f"{output_dir}/validation_features.csv", index=False)
    test.to_csv(f"{output_dir}/test_features.csv", index=False)
    print("Feature engineering done.")

if __name__ == "__main__":
    engineer_features(
        "../data/processed",
        "../data/raw/ERA5_Weather_Data_Monash.csv",
        "../data/processed"
    )