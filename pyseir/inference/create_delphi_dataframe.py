import math
import us
from datetime import datetime, timedelta
import numpy as np
import logging
import pandas as pd
import os, sys, glob
from matplotlib import pyplot as plt
import us
import structlog

CSV_FOLDER = "/Users/natashawoods/Desktop/later.nosync/covid_act_now.nosync/covid-data-public/data/aws-lake/old_aws_lake/"

# Get list of all available CSV files
csv_files = glob.glob(CSV_FOLDER + "*csv")


# Load google health trends data
ght = pd.read_csv(
    CSV_FOLDER + "timeseries-ght.csv", converters={"fips": str}, parse_dates=True, index_col="date"
)
aggregate_level_name = "aggregate_level"
aggregate_select = "state"
ght = ght[ght[aggregate_level_name] == aggregate_select]
ght.to_csv("ght_test.csv")


# load CAN input data
can_data = "/Users/natashawoods/Desktop/later.nosync/covid_act_now.nosync/covid-data-model/pyseir_data/merged_results.csv"
can_df = pd.read_csv(can_data, converters={"fips": str}, parse_dates=True, index_col="date")

# Merge dataframes
merged_df = pd.merge(
    can_df,
    ght,
    how="left",
    left_on=["fips", "date", "aggregate_level", "state"],
    right_on=["fips", "date", "aggregate_level", "state"],
)
merged_df.to_csv("delphi_merged.csv")


print(ght)

print(csv_files)
