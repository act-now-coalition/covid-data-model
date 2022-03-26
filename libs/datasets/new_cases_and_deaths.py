from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields
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
    series: pd.Series, max_days_to_spread: int = 14, is_multiindex: bool = False
) -> pd.Series:
    """Spreads first reported value after reported zeros by dividing it evenly
    over the prior days.

    Args:
        series: Series of new cases with date index.
        max_days_to_spread: Maximum number of days to spread a single report. If
            a stall exceeds this number of days, the reported value will be spread
            over max_days_to_spread and the remaining zeros will be kept as
            zeros.
        is_multiindex: Whether or not the series contains a multi-level index. If
            true, one level must be a date index.
    """
    if not (series > 0).any():
        return series

    # Remove NaN values from backfill calculations.
    # We will re-add the NaN indices at the end,
    # this way NaN values do not have cases spread to them and they do not reset stalled_days_count.
    # NaNs are created from data blocked through manual region overrides and outlier detection.
    empty_dates = series[series.isna()]
    series = series.dropna()

    # Counting consecutive zeros
    zeros = series == 0
    zeros_count = zeros.cumsum()

    stalled_days_count = zeros_count.sub(zeros_count.mask(zeros).ffill().fillna(0))

    # Add one more day for spreading on first report after zeros
    first_report_after_zeros = (series != 0) & (series.shift(1) == 0)
    stalled_days_count = stalled_days_count + (
        (stalled_days_count.shift(1).fillna(0) + 1) * first_report_after_zeros
    )
    num_days_worth_of_cases = stalled_days_count * first_report_after_zeros

    # Backfill number of days to spread to preceding zeros
    temp = num_days_worth_of_cases.copy()
    temp[series == 0] = None
    bfilled_num_days = temp.bfill()

    # Find zeros that are within the boundary of `max_days_to_spread`
    zeros_to_keep = (bfilled_num_days - stalled_days_count) >= max_days_to_spread
    zeros_to_replace = ~zeros_to_keep & (series == 0)

    # Calculate spreading factor (clipping at max_days_to_spread)
    num_days_worth_of_cases = num_days_worth_of_cases.clip(0, max_days_to_spread)
    num_days_worth_of_cases[~first_report_after_zeros] = 1
    num_days_worth_of_cases[zeros_to_replace] = None

    # Don't mess with leading / trailing zeros.
    if is_multiindex:
        non_zero_date_index = series[series.gt(0)].index.get_level_values("date")
        first_case = non_zero_date_index[0]
        last_case = non_zero_date_index[-1]
        date_index = series.index.get_level_values("date")
        is_between_first_last_case = (date_index > first_case) & (date_index <= last_case)
    else:
        first_case = series[series.gt(0)].index[0]
        last_case = series[series.gt(0)].index[-1]
        is_between_first_last_case = (series.index > first_case) & (series.index <= last_case)

    series[is_between_first_last_case] = (
        series[is_between_first_last_case] / num_days_worth_of_cases
    )

    zeros_to_replace = zeros_to_replace & is_between_first_last_case
    series[zeros_to_replace] = None
    series[is_between_first_last_case] = series[is_between_first_last_case].bfill()

    # Re-insert NaN values into the timeseries in their proper locations
    return series.combine_first(empty_dates)
