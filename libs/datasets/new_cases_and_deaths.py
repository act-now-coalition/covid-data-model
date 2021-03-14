from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
from libs.datasets import timeseries

import pandas as pd


MultiRegionDataset = timeseries.MultiRegionDataset


def _diff_preserving_first_value(
    series: pd.Series, field_in: CommonFields, field_out: CommonFields
):
    cases = series.reset_index(CommonFields.LOCATION_ID, drop=True).loc[field_in, :]
    # cases is a pd.Series (a 1-D vector) with DATE index
    assert cases.index.names == [CommonFields.DATE]
    new_cases = cases.diff()
    first_date = cases.notna().idxmax()
    if pd.notna(first_date):
        new_cases[first_date] = cases[first_date]

    # Return a DataFrame so NEW_CASES is a column with DATE index.
    return pd.DataFrame({field_out: new_cases})


def add_incident_column(
    dataset_in: MultiRegionDataset, field_in: CommonFields, field_out: CommonFields
) -> MultiRegionDataset:

    # Get timeseries data from timeseries_wide_dates because it creates a date range that includes
    # every date, even those with NA cases. This keeps the output identical when empty rows are
    # dropped or added.
    wide_dates_var = dataset_in.timeseries_wide_dates_no_buckets().loc[(slice(None), field_in), :]
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # the first day as appropriate new case data.
    # We want as_index=True so that the DataFrame returned by each _diff_preserving_first_value call
    # has the location_id added as an index before being concat-ed.
    new_cases = (
        wide_dates_var.groupby(CommonFields.LOCATION_ID, as_index=True, sort=False)
        .apply(lambda x: _diff_preserving_first_value(x, field_in, field_out))
        .rename_axis(columns=PdFields.VARIABLE)
    )

    # Replacing days with single back tracking adjustments to be 0, reduces
    # number of na days in timeseries
    new_cases[new_cases == -1] = 0

    # Remove the occasional negative case adjustments.
    # TODO: Add annotation
    new_cases[new_cases < 0] = pd.NA
    new_cases = new_cases.dropna()

    new_cases_dataset = MultiRegionDataset(timeseries=new_cases)

    dataset_out = dataset_in.join_columns(new_cases_dataset)
    return dataset_out


def add_new_cases(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""

    return add_incident_column(dataset_in, CommonFields.CASES, CommonFields.NEW_CASES)


def add_new_deaths(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""

    return add_incident_column(dataset_in, CommonFields.DEATHS, CommonFields.NEW_DEATHS)
