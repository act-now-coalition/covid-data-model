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
from functools import reduce

CSV_FOLDER = (
    "/Users/natashawoods/Desktop/later.nosync/covid_act_now.nosync/covid-data-public/data/aws-lake/"
)
aggregate_level_name = "aggregate_level"
aggregate_select = "state"

# load CAN input data
can_data = "/Users/natashawoods/Desktop/later.nosync/covid_act_now.nosync/covid-data-model/pyseir_data/merged_results.csv"
can_df = pd.read_csv(can_data, converters={"fips": str}, parse_dates=True, index_col="date")

# Get list of all available CSV files
csv_files = glob.glob(CSV_FOLDER + "*csv")
print(csv_files)

delphi_dataframes = []
for delphi_file in csv_files:
    delphi_var_df = pd.read_csv(
        delphi_file, converters={"fips": str}, parse_dates=True, index_col="date"
    )

    delphi_var_df = delphi_var_df[delphi_var_df[aggregate_level_name] == aggregate_select]

    delphi_dataframes.append(delphi_var_df)

merged_df = reduce(
    lambda left, right: pd.merge(
        left,
        right,
        left_on=["fips", "date", aggregate_level_name, aggregate_select],
        right_on=["fips", "date", aggregate_level_name, aggregate_select],
    ),
    delphi_dataframes,
)

merged_df.to_csv("merged_delphi_df.csv")
# Merge dataframes
# merged_df = pd.merge(
#    can_df,
#    delphi_var_df,
#    how="left",
#    left_on=["fips", "date", aggregate_level_name, aggregate_select],
#    right_on=["fips", "date", aggregate_level_name, aggregate_select],
# )

# merged_df["fips_int"] = merged_df["fips"].astype(int)
# merged_df.to_csv("delphi_merged.csv")
