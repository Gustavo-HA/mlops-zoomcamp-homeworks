import pickle
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression

import mlflow
from prefect import flow, task

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-yellow-taxi-duration-prediction")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@task(name="Reading dataframe",log_prints=True)
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    print(f"Read {df.shape[0]} rows from {url}")
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"After filtering, {df.shape[0]} rows remain")
    return df

@task(name="Creating feature matrix", log_prints=True)
def create_X(df : pd.DataFrame, dv=None) -> Tuple[Any, DictVectorizer]:
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return (X, dv)

@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val, dv) -> str:
    with mlflow.start_run() as run:
        train = X_train
        valid = y_train

        model = LinearRegression()
        model.fit(train, valid)
        print(f"Intercept: {model.intercept_}")
        
        y_pred = model.predict(X_val)
        
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id

@flow(name="Run duration prediction training")
def run(year, month) -> str:
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month > 1 else year - 1
    next_month = month - 1 if month > 1 else 12
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    run_id = str(run_id)

    with open("run_id.txt", "w") as f:
        f.write(run_id)