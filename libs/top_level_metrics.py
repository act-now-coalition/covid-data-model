import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import combined_datasets


def calculate_top_level_metrics_for_fips(fips: str):
    timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = combined_datasets.load_us_latest_dataset()

    fips_timeseries = timeseries.get_subset(fips=fips)
    fips_record = latest.get_record_for_fips(fips)

    # not sure of return type for now, could be a dictionary, or maybe it would be more effective
    # as a pandas dataframe with a column for each metric.
    return calculate_top_level_metrics_for_timeseries(fips_timeseries, fips_record)


def calculate_top_level_metrics_for_timeseries(timeseries: TimeseriesDataset, latest: dict):
    # Making sure that the timeseries object passed in is only for one fips.
    assert len(timeseries.all_fips) == 1
    population = latest[CommonFields.POPULATION]
    data = timeseries.data

    case_density = calculate_case_density(timeseries.data[CommonFields.CASES], population)
    test_positivity = calculate_test_positivity()

    return {"case_density": case_density, "test_positivity": test_positivity}


def calculate_case_density(cases: pd.Series, population: int) -> pd.Series:
    # Current calculation for this occurs here:
    # https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L448-L457
    # You'll probably have to jump around in the code to figure out the various smoothing
    # applied.
    pass


def calculate_test_positivity():
    # https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L506-L533
    pass


# Example of running calculation for all counties in a state, using the latest dataset
# to get all fips codes for that state
def calculate_metrics_for_counties_in_state(state: str):
    latest = combined_datasets.load_us_latest_dataset()
    state_latest_values = latest.county.get_subset(state=state)
    for fips in state_latest_values.all_fips:
        yield calculate_top_level_metrics_for_fips(fips)
