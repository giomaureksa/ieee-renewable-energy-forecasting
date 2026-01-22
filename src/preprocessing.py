import pandas as pd
from datetime import timedelta

def parse_tsf(path, freq_minutes=15):
    records = []
    in_data_section = False
    
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("@data"):
                in_data_section = True
                continue
            if not in_data_section:
                continue
            if line.startswith("@"):
                continue
            try:
                series_id, rest = line.split(":", 1)
            except ValueError:
                continue
            values = rest.split(",")
            start_time = pd.to_datetime(values[0], errors="coerce")
            if pd.isna(start_time):
                continue
            for i, v in enumerate(values[1:]):
                try:
                    val = float(v)
                except ValueError:
                    continue
                records.append({
                    "series_id": series_id.strip(),
                    "timestamp": start_time + timedelta(minutes=freq_minutes * i),
                    "value": val
                })
    return pd.DataFrame(records)

def normalize_timestamp(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    return df

def preprocess_data(raw_dir, processed_dir):
    print("Loading raw data...")
    train = parse_tsf(f"{raw_dir}/train.csv")
    val = parse_tsf(f"{raw_dir}/validation.csv")
    test = parse_tsf(f"{raw_dir}/test.csv")
    
    print("Normalizing timestamps...")
    train = normalize_timestamp(train)
    val = normalize_timestamp(val)
    test = normalize_timestamp(test)
    
    print("Saving processed data...")
    train.to_csv(f"{processed_dir}/train_processed.csv", index=False)
    val.to_csv(f"{processed_dir}/validation_processed.csv", index=False)
    test.to_csv(f"{processed_dir}/test_processed.csv", index=False)
    print("Preprocessing done.")

if __name__ == "__main__":
    preprocess_data("../data/raw", "../data/processed")