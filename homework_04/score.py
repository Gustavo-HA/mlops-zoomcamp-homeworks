import os
import sys
import pickle
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

def read_data(filename, year, month, CATEGORICAL):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    
    # df["ride_id"] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def prepare_dictionaries(df : pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts
    
def apply_model(df, dicts, dv, model, year, month):
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Std of predictions: {y_pred.std():.4f}")
    print(f"Mean of predictions: {y_pred.mean():.4f}")

    df["ride_id"] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({"ride_id": df.ride_id, "prediction": y_pred})
    
    return df_result

def run():
    taxi_type = sys.argv[1]
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    
    INPUT_FILE = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    OUTPUT_FILE = f"./output/{taxi_type}/{year:04d}-{month:02d}.parquet"
    CATEGORICAL = ['PULocationID', 'DOLocationID']
    
    print(f"Reading data from {INPUT_FILE} for {year:04d}-{month:02d}...")
    df = read_data(INPUT_FILE, year, month, CATEGORICAL)
    
    print(f"Loading model from model.bin...")
    dv, model = load_model()
    
    print(f"Preparing dictionaries for DictVectorizer...")
    dicts = prepare_dictionaries(df)
    
    print(f"Applying model to the data...")
    df_result = apply_model(df, dicts, dv, model, year, month)
    
    print(f"Saving results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_result.to_parquet(
        OUTPUT_FILE, 
        engine='pyarrow', 
        compression=None, 
        index=False
    )
    
    print(f"{Path(OUTPUT_FILE).stat().st_size >> 20} MB written to {OUTPUT_FILE}.")
    
    print("Done.")

if __name__ == "__main__":
    run()