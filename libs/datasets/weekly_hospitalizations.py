from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields
from libs.datasets import timeseries

import pandas as pd


MultiRegionDataset = timeseries.MultiRegionDataset


def _rolling_avg_7day(daily: pd.Series):
    # cases is a pd.Series (a 1-D vector) with DATE index
    assert daily.index.names == [CommonFields.DATE]
    return daily.rolling(7)


def add_weekly_rolling_average_column(
    dataset_in: MultiRegionDataset, field_in: CommonFields, field_out: CommonFields
) -> MultiRegionDataset:
    assert field_out not in dataset_in.timeseries_bucketed_wide_dates.index.unique(
        PdFields.VARIABLE
    )

    # Get timeseries data from timeseries_wide_dates because it creates a date range that includes
    # every date, even those with NA values. This keeps the output identical when empty rows are
    # dropped or added.
    wide_dates_var = dataset_in.timeseries_bucketed_wide_dates.xs(
        field_in, level=PdFields.VARIABLE, drop_level=False
    )
    # We want as_index=True so that the DataFrame returned by each _diff_preserving_first_value call
    # has the location_id added as an index before being concat-ed.
    weekly_admissions = wide_dates_var.apply(
        _rolling_avg_7day, axis=1, result_type="reduce"
    ).rename({field_in: field_out}, axis="index", level=PdFields.VARIABLE)

    # TODO: Add annotation
    # Drop time-series (rows) that don't have any real values.
    weekly_admissions = weekly_admissions.dropna(axis="index", how="all")

    weekly_admissions_dataset = MultiRegionDataset.from_timeseries_wide_dates_df(
        weekly_admissions, bucketed=True
    )

    dataset_out = dataset_in.join_columns(weekly_admissions_dataset)
    return dataset_out


def add_weekly_hospitalizations(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a weekly_hospitalizations column to this dataset by calculating the rolling average of ."""

    return add_weekly_rolling_average_column(
        dataset_in,
        CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID,
        CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID,
    )
