from evidently import Report
from evidently.ui.workspace import Workspace, Project
from evidently.sdk.panels import *
from evidently.sdk.models import PanelMetric
from evidently.legacy.renderers.html_widgets import WidgetSize
from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.metrics import (
    ValueDrift, 
    DriftedColumnsCount,
    MissingValueCount,
    StdValue,
    QuantileValue
    )
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
import os
from typing import Tuple


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the reference and current data from parquet files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the reference and current data DataFrames.
    """
    reference_data = pd.read_parquet(os.getenv("ev_reference_data_path"))
    current_data = pd.read_parquet(os.getenv("ev_current_data_path"))
    return reference_data, current_data

def create_daily_reports_by_month(data_definition : DataDefinition,
                                  reference_data : pd.DataFrame,
                                  current_data : pd.DataFrame,
                                  year : int = 2024,
                                  month : int = 3) -> Project:
    """Creates daily reports for the specified month and year using the provided data definition, reference data, and current data."""
    
    ws = Workspace("workspace")
    project_id = os.getenv("ev_project_id")
    project = ws.get_project(project_id=project_id)

    report = Report(metrics=[
        StdValue(column='prediction'),
        QuantileValue(column='fare_amount', quantile=0.5),
        DriftedColumnsCount(),
    ]
    )

    # Report by day
    for i in range(1,29):
        print(f"Processing data for {year}-{month:02d}-{i:02d}")
        
        if len(current_data.loc[current_data.lpep_pickup_datetime.between(
            f'{year}-{month:02d}-{i:02d}', 
            f'{year}-{month:02d}-{i+1:02d}', 
            inclusive="left")]) == 0:
            print(f"No data for {year}-{month:02d}-{i:02d}")
            continue
        
        day_data = current_data.loc[current_data.lpep_pickup_datetime.between(
            f'{year}-{month:02d}-{i:02d}', 
            f'{year}-{month:02d}-{i+1:02d}', 
            inclusive="left")]

        data = Dataset.from_pandas(data=day_data,
                                data_definition = data)

        snapshot = report.run(reference_data=reference_data,
                    current_data=data,
                    timestamp=dt.datetime(2024, 3, i))
        
        ws.add_run(project.id, snapshot)

    return project

def create_dashboard(project) -> None:
    project.dashboard.clear_dashboard()
    
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Homework 5 - NYC Taxi Data",
            size="full",
            values=[],
            plot_params={"plot_type": "text"}
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Prediction Standard Deviation",
            size="half",
            values=[
                PanelMetric(legend="std", metric="StdValue", metric_labels={"column":"prediction"})
            ]
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Median Fare Amount",
            size="half",
            values=[
                PanelMetric(legend="median", metric="QuantileValue", metric_labels={"column":"fare_amount", "quantile":0.5})
            ]
        )
    )


def run(year, month) -> None:
    """
    Main function to run the monitoring process.
    
    Args:
        year (int): The year for which the reports are generated.
        month (int): The month for which the reports are generated.
    """

    load_dotenv()

    reference_data, current_data = read_data()

    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    data_definition = DataDefinition(numerical_columns = num_features + ['prediction'],
                                    categorical_columns=cat_features)

    project = create_daily_reports_by_month(data_definition=data_definition,
                                reference_data=reference_data,
                                current_data=current_data,
                                year=year,
                                month=month)

    create_dashboard(project)

if __name__ == "__main__":
    run(year=2024, month=3) # todo with click/argparse