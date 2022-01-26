"""
Compare two dataframes loaded from CSV files. The files must have a single key column, currently
named `location_id` and a single time column named `date`. All other columns are treated as
timeseries
points to be compared. The column name of timeseries points is pivoted/stacked/melted to the
`variable` column. A timeseries is the sequence of points from the same original column, with the
same location_id and is identified by the <location_id, variable> pair. A file is said to contain a
timeseries with identity <location_id, variable> if the melted representation contains it
contains at least one row for the <location_id, variable> and a real (not NaN) value.

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

from datapublic.common_fields import CommonFields, PdFields


TIMESERIES_KEYS = [CommonFields.LOCATION_ID, CommonFields.DATE]


class DatasetDiff(BaseModel):
    # Duplicates that are dropped from consideration for the diff
    duplicates_dropped: pd.DataFrame
    # The non-duplicated values pivoted/stacked/melted with new column 'value' and new index level
    # 'variable'
    melt: pd.DataFrame
    # MultiIndex with levels 'variable' and 'location_id'. Since a timeseries in a dataset is
    # identified
    # by a <variable, location_id> pair this is usable as the set of all timeseries in this dataset.
    all_variable_location_id: pd.MultiIndex
    # The subset of all_variable_location_id that appears in this dataset but not the other one
    my_ts: Optional[pd.MultiIndex] = None
    # The timeseries that appear in both this dataset and the other one
    common_location_id: Optional[pd.DataFrame] = None
    # Timeseries points that appear in this dataset but not the other one
    my_ts_points: Optional[pd.DataFrame] = None
    # Diffs of overlapping parts of the timeseries, only set on the left dataset
    ts_diffs: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        # TODO(tom): Make the output easier to read. See idea in docstring at top of this file.
        if self.ts_diffs is None or self.ts_diffs.empty:
            ts_diffs_str = ""
        else:
            most_diff_timeseries = self.ts_diffs.sort_values("diff", ascending=False).head(20)
            most_diff_variables = (
                self.ts_diffs.groupby("variable has_overlap".split())
                .mean()
                .sort_values("diff", ascending=False)
                .head(20)
            )
            ts_diffs_str = f"""TS diffs:\n{most_diff_timeseries}
TS diffs by variable and has_overlap:\n{most_diff_variables}
"""

        return f"""Duplicate rows in this file: {self.duplicates_dropped.index.unique(level='location_id')}
{self.duplicates_dropped}
TS only in this file: {self.my_ts}
TS points only in this file: {self.my_ts_points.groupby('date').size().to_dict()}
{ts_diffs_str}
"""

    @staticmethod
    def make(df: pd.DataFrame) -> "DatasetDiff":
        dups_bool_array = df.index.duplicated(keep=False)
        dups = df.loc[dups_bool_array, :]
        df = df.loc[~dups_bool_array, :]

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
            if col != CommonFields.LOCATION_ID:
                print(f"Dropping field '{col}' based on type")
                columns_to_drop.add(col)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        melt = (
            df.melt(id_vars=TIMESERIES_KEYS)
            .set_index([PdFields.VARIABLE] + TIMESERIES_KEYS)
            .dropna()
        )
        # When Int64 and float64 columns are merged into one by melt the 'object' dtype is used, which
        # is not supported by timeseries_diff. Force 'value' back to a numeric dtype.
        melt["value"] = pd.to_numeric(melt["value"])

        all_variable_locations = (
            melt.groupby([PdFields.VARIABLE, CommonFields.LOCATION_ID]).first().index
        )
        return DatasetDiff(
            duplicates_dropped=dups, melt=melt, all_variable_location_id=all_variable_locations
        )

    def compare(self, other: "DatasetDiff"):
        self.my_ts = self.all_variable_location_id.difference(other.all_variable_location_id)
        other.my_ts = other.all_variable_location_id.difference(self.all_variable_location_id)

        # The rest of this function works with timeseries that have at least one real value in both
        # original dataframes.
        common_variable_location_id = self.all_variable_location_id.intersection(
            other.all_variable_location_id
        )
        self.common_location_id = self.melt.loc[
            self.melt.reset_index(CommonFields.DATE).index.isin(common_variable_location_id)
        ]
        other.common_location_id = other.melt.loc[
            other.melt.reset_index(CommonFields.DATE).index.isin(common_variable_location_id)
        ]

        joined_ts = pd.merge(
            self.common_location_id["value"],
            other.common_location_id["value"],
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
        self.ts_diffs = joined_ts.groupby([PdFields.VARIABLE, CommonFields.LOCATION_ID]).apply(
            timeseries_diff
        )


def timeseries_diff(group_subframe: pd.DataFrame) -> pd.Series:
    try:
        ts = group_subframe.droplevel([PdFields.VARIABLE, CommonFields.LOCATION_ID])
        right = ts["value_r"]
        left = ts["value_l"]
        # Ignoring gaps of NaN between real values, find the longest range of dates where the right
        # and left overlap.
        start = max(right.first_valid_index(), left.first_valid_index())
        end = min(right.last_valid_index(), left.last_valid_index())

        if start <= end:
            right_common_ts = right.loc[start:end].interpolate(method="time")
            left_common_ts = left.loc[start:end].interpolate(method="time")
            # Sum before divide suggest by formula for SMAPE at
            # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
            common_sum = right_common_ts.sum() + left_common_ts.sum()
            common_abs_diff = (right_common_ts - left_common_ts).abs().sum()
            if abs(common_sum) > 0.0001:
                diff = common_abs_diff / common_sum
            else:
                # Hack to avoid dividing by a tiny number (or 0, bomb!) when common_sum is small.
                diff = 0.0 if abs(common_abs_diff) < 0.0001 else 1.0

            # if diff > 0.04:
            #    print(f"diff over 0.04")
            #    print(group_subframe)
            #    print(f"from {start} to {end}")
            #    print(f"common sum {common_sum} and diff {common_abs_diff}")
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
