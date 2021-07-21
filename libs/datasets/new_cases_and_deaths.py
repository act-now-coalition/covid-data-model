from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
from libs.datasets import timeseries

import pandas as pd


MultiRegionDataset = timeseries.MultiRegionDataset


# TODO(michael): We used to apply special logic to keep the first date's cases
# as new_cases, but we have removed that.  This function could be removed.
def _diff_preserving_first_value(cases: pd.Series):
    # cases is a pd.Series (a 1-D vector) with DATE index
    assert cases.index.names == [CommonFields.DATE]
    new_cases = cases.diff()
    return new_cases


def add_incident_column(
    dataset_in: MultiRegionDataset, field_in: CommonFields, field_out: CommonFields
) -> MultiRegionDataset:
    assert field_out not in dataset_in.timeseries_bucketed_wide_dates.index.unique(
        PdFields.VARIABLE
    )

    # Get timeseries data from timeseries_wide_dates because it creates a date range that includes
    # every date, even those with NA cases. This keeps the output identical when empty rows are
    # dropped or added.
    wide_dates_var = dataset_in.timeseries_bucketed_wide_dates.xs(
        field_in, level=PdFields.VARIABLE, drop_level=False
    )
    # TODO(michael): This comment is no longer accurate. See _diff_preserving_first_value().
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # the first day as appropriate new case data.
    # We want as_index=True so that the DataFrame returned by each _diff_preserving_first_value call
    # has the location_id added as an index before being concat-ed.
    new_cases = wide_dates_var.apply(
        _diff_preserving_first_value, axis=1, result_type="reduce"
    ).rename({field_in: field_out}, axis="index", level=PdFields.VARIABLE)

    # Replacing days with single back tracking adjustments to be 0, reduces
    # number of na days in timeseries
    new_cases[new_cases == -1] = 0

    # Remove the occasional negative case adjustments.
    # TODO: Add annotation
    new_cases[new_cases < 0] = pd.NA
    # Drop time-series (rows) that don't have any real values.
    new_cases = new_cases.dropna(axis="index", how="all")

    new_cases_dataset = MultiRegionDataset.from_timeseries_wide_dates_df(new_cases, bucketed=True)

    dataset_out = dataset_in.join_columns(new_cases_dataset)
    return dataset_out


def add_new_cases(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""

    return add_incident_column(dataset_in, CommonFields.CASES, CommonFields.NEW_CASES)


def add_new_deaths(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""

    return add_incident_column(dataset_in, CommonFields.DEATHS, CommonFields.NEW_DEATHS)


def spread_first_reported_value_after_stall(
    series: pd.Series, max_days_to_spread: int = 7
) -> pd.Series:
    """Spreads first reported value after reported zeros.

    TODO(chris): Make a max number of days.

    Args:
        series: Series of new cases with date index.
    """
    # Find points in the series that are either zeros or the first report after a string of zeros.

    zeros_or_first_report = (series == 0) | (series.shift(1) == 0)
    zeros_or_first_report_count = zeros_or_first_report.cumsum()
    first_report_after_zeros = (series != 0) & (series.shift(1) == 0)

    first_report_after_zeros = (series != 0) & (series.shift(1) == 0)

    stalled_cases_count = zeros_or_first_report_count.sub(
        zeros_or_first_report_count.mask(zeros_or_first_report).ffill().fillna(0)
    )

    num_days_worth_of_cases = stalled_cases_count * first_report_after_zeros

    temp = num_days_worth_of_cases.copy()
    temp[series == 0] = None
    bfilled_num_days = temp.bfill()

    zeros_to_keep = ((bfilled_num_days - stalled_cases_count) >= max_days_to_spread) & (series == 0)
    zeros_to_replace = ~zeros_to_keep & (series == 0)

    num_days_worth_of_cases = num_days_worth_of_cases.clip(0, max_days_to_spread)
    num_days_worth_of_cases[~first_report_after_zeros] = 1
    num_days_worth_of_cases[zeros_to_replace] = None

    is_after_first_case = series.index > series.first_valid_index()

    series[is_after_first_case] = series[is_after_first_case] / num_days_worth_of_cases

    series[zeros_to_replace] = None
    series[is_after_first_case] = series[is_after_first_case].bfill()

    return series
