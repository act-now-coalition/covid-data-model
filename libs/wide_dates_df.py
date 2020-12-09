"""
Functions that process pd.DataFrame objects with a column for each date.

These DataFrame objects are generally created by `MultiRegionDataset.timeseries_wide_dates()`.
"""


import pathlib
import pandas as pd


def write_csv(wide_df: pd.DataFrame, path: pathlib.Path):
    """Writes a DataFrame with date columns, as produced by MultiRegionDataset.timeseries_rows()"""
    wide_df.to_csv(
        path, date_format="%Y-%m-%d", index=True, float_format="%.12g",
    )
