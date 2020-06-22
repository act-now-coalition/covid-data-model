"""
Compare two dataframes loaded from CSV files. The files must have a single key column, currently
named `fips` and a single time column named `date`. All other columns are treated as timeseries
points to be compared. The column name of timeseries points is pivoted/stacked/melted to the
`variable` column. A timeseries is the sequence of points from the same original column, with the
same fips and is identified by the <fips, variable> pair. A file is said to contain a timeseries
with identity <fips, variable> if the melted representation contains it contains at least one
row for the <fips, variable> and a real (not NaN) value.

This tool helps us find differences between files by aggregating the sets of timeseries in the files
and outputting the differences.

First timeseries points that appear in the same file more than once are dropped and output. A
summary of the set of timeseries that appear in exactly one file is output. Then the common
timeseries are compared. A summary of timeseries points that appear in exactly one file are output.
sections of the timeseries that appear in both files are compared and the amount of difference is
aggregated and output.


Proposed output per variable:

Variable: 'cases'

Duplicate TS points dropped:
    Left: FIPS xxyyy, xxyyy
    Right: FIPS xxyyy, xxyyy
Missing TS:
    Left only: FIPS xxyyy, xxyyy
    Right only: FIPS xxyyy, xxyyy
TS Points Missing:
    Left only Jun 21: FIPS xxyyy, xxyyy
    Left only Jun 20: FIPS xxyyy, xxyyy
    Left only Jun 19: FIPS xxyyy, xxyyy
    Left only Jun 18-11: FIPS xxyyy, xxyyy
    Left only Jun 10 - May 10: FIPS xxyyy, xxyyy
Common TS:
    No time range overlap: FIPS xxyyyy, ...
    1-10 point overlap: most diff 0.4 FIPS xxyyy, median diff 0.2 FIPS xxyyy, least diff 0.01 FIPS xxyyy
    11-21 point overlap: most diff 0.4 FIPS xxyyy, median diff 0.2 FIPS xxyyy, least diff 0.01 FIPS xxyyy


Proposed way to print a bunch of FIPS:
FIPS 5 states (xx, yy, zz, ...) and 6 counties (xxyyy, xxyyy, ... OR by state TX: 10, TN 10, ...)

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
        df[CommonFields.DATE] = pd.to_datetime(df[CommonFields.DATE])
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
                print(f"Dropping field '{col}' based on type")
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
