"""
CAUTION: UNREVIEWED BETA CODE

Compare two dataframes loaded from CSV files. The files must have a single key column, currently
named `fips` and a single time column named `date`. All other columns are treated as timeseries
values to be compared.
"""


from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel

from covidactnow.datapublic.common_fields import CommonFields, COMMON_FIELDS_TIMESERIES_KEYS


class DatasetDiff(BaseModel):
    duplicates_dropped: pd.DataFrame
    melt: pd.DataFrame
    all_variable_fips: pd.MultiIndex
    my_ts: Optional[pd.MultiIndex] = None
    common_fips: Optional[pd.DataFrame] = None
    my_ts_points: Optional[pd.DataFrame] = None
    ts_diffs: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"""Duplicate rows in this file: {self.duplicates_dropped}
TS only in this file: {self.my_ts}
TS points only in this file: {self.my_ts_points.groupby('date').size().to_dict()}
TS diffs: {self.ts_diffs if self.ts_diffs is not None else ''}
TS diffs: {self.ts_diffs.groupby('variable has_overlap'.split()).mean() if self.ts_diffs is not None else ''}
"""

    @staticmethod
    def make(df: pd.DataFrame) -> "DatasetDiff":
        dups = df.loc[df.index.duplicated(keep=False)]
        if not dups.empty:
            df = df.drop_duplicates()

        df = df.reset_index().replace({pd.NA: np.nan}).convert_dtypes()
        # Drop columns that don't really contain timeseries values.
        columns_to_drop = {
            "index",
            CommonFields.STATE,
            CommonFields.COUNTRY,
            CommonFields.AGGREGATE_LEVEL,
        }.intersection(df.columns)
        # Drop string columns because timeseries_diff can't handle them yet.
        for col in df.select_dtypes(include={"object", "string"}):
            if col != "fips":
                print(f"dropping based on type {col}")
                columns_to_drop.add(col)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        melt = (
            df.melt(id_vars=COMMON_FIELDS_TIMESERIES_KEYS)
            .set_index(["variable"] + COMMON_FIELDS_TIMESERIES_KEYS)
            .dropna()
        )
        # When Int64 and float64 columns are merged into one by melt the 'object' dtype is used, which
        # is not supported by timeseries_diff. Force 'value' back to a numeric dtype.
        melt["value"] = pd.to_numeric(melt["value"])

        all_variable_fips = melt.groupby("variable fips".split()).first().index
        return DatasetDiff(duplicates_dropped=dups, melt=melt, all_variable_fips=all_variable_fips)

    def compare(self, other: "DatasetDiff"):
        # A timeseries is a set of <date, value> tuples identified by a <variable, fips> tuple. First
        # find all the timeseries that have at least one real value in only one dataset. These are
        # timeseries that appear in one file but not the other file.
        self.my_ts = self.all_variable_fips.difference(other.all_variable_fips)
        other.my_ts = other.all_variable_fips.difference(self.all_variable_fips)

        # The rest of this function works with timeseries that have at least one real value in both
        # original dataframes.
        common_variable_fips = self.all_variable_fips.intersection(other.all_variable_fips)
        self.common_fips = self.melt.loc[
            self.melt.reset_index(CommonFields.DATE).index.isin(common_variable_fips)
        ]
        other.common_fips = other.melt.loc[
            other.melt.reset_index(CommonFields.DATE).index.isin(common_variable_fips)
        ]

        joined_ts = pd.merge(
            self.common_fips["value"],
            other.common_fips["value"],
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("_l", "_r"),
        )
        joined_ts_notna = joined_ts.notna()
        # Among the common timeseries, find points that appear in exactly one dataset.
        self.my_ts_points = joined_ts.loc[
            joined_ts_notna["value_l"] & ~joined_ts_notna["value_r"], "value_l"
        ]
        other.my_ts_points = joined_ts.loc[
            joined_ts_notna["value_r"] & ~joined_ts_notna["value_l"], "value_r"
        ]
        # Find some kind of measure of difference between each common timeseries.
        self.ts_diffs = joined_ts.groupby("variable fips".split()).apply(timeseries_diff)


def timeseries_diff(ts: pd.DataFrame) -> pd.Series:
    try:
        ts = ts.droplevel(["variable", CommonFields.FIPS])
        right = ts["value_r"]
        left = ts["value_l"]
        # Ignoring gaps of NaN between real values, find the longest range of dates where the right
        # and left overlap.
        start = max(right.idxmin(), left.idxmin())
        end = min(right.idxmax(), left.idxmax())
        if start <= end:
            right_common_ts = right.loc[start:end].interpolate(method="time")
            left_common_ts = left.loc[start:end].interpolate(method="time")
            diff = (
                (right_common_ts - left_common_ts).abs() / ((right_common_ts + left_common_ts) / 2)
            ).mean()
            # if diff > 0.01:
            #    print(ts)
            #    print(f"from {start} to {end}")
            rv = pd.Series(
                [diff, len(right_common_ts), True], index=["diff", "points_overlap", "has_overlap"]
            )
            return rv
        else:
            return pd.Series([1.0, 0, False], index=["diff", "points_overlap", "has_overlap"])
    except:
        ts.info()
        print(ts)
        raise
