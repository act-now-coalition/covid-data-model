"""
Functions that process pd.DataFrame objects with a column for each date.

These DataFrame objects are generally created by `TimeseriesDataset.get_date_columns`.
"""


import pathlib
import pandas as pd


# TODO(tom): The output currently has 3 rows of headers and includes the time with dates. Change the output into
# something that is reasonable to parse and write a function to read it.
def write_csv(wide_df: pd.DataFrame, path: pathlib.Path):
    wide_df.to_csv(
        path, date_format="%Y-%m-%d", index=True, float_format="%.12g",
    )
