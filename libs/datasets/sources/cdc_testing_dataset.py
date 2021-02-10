from libs.datasets import data_source
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


DC_COUNTY_FIPS = "11001"
DC_STATE_FIPS = "11"


def _remove_trailing_zeros(series: pd.Series) -> pd.Series:

    series = series.copy()

    index = series.loc[series != 0].last_valid_index()

    if index is None:
        # If test positivity is 0% the entire time, considering the data inaccurate, returning
        # none.
        series[:] = None
        return series

    series[index + pd.DateOffset(1) :] = None
    return series


def remove_trailing_zeros(data: pd.DataFrame) -> pd.DataFrame:
    # TODO(tom): See if TailFilter+zeros_filter produce the same data and if so, remove this
    #  function.
    data = data.sort_values([CommonFields.FIPS, CommonFields.DATE]).set_index(CommonFields.DATE)
    test_pos = data.groupby(CommonFields.FIPS)[CommonFields.TEST_POSITIVITY_7D].apply(
        _remove_trailing_zeros
    )
    data[CommonFields.TEST_POSITIVITY_7D] = test_pos
    return data.reset_index()


def transform(dataset: ccd_helpers.CanScraperLoader):

    variables = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="rolling_average_7_day",
            provider="cdc",
            unit="percentage",
            common_field=CommonFields.TEST_POSITIVITY_7D,
        ),
    ]
    results = dataset.query_multiple_variables(variables)
    # Test positivity should be a ratio
    results.loc[:, CommonFields.TEST_POSITIVITY_7D] = (
        results.loc[:, CommonFields.TEST_POSITIVITY_7D] / 100.0
    )
    # Should only be picking up county all_df for now.  May need additional logic if states
    # are included as well
    assert (results[CommonFields.FIPS].str.len() == 5).all()

    # Duplicating DC County results as state results because of a downstream
    # use of how dc state data is used to override DC county data.
    dc_results = results.loc[results[CommonFields.FIPS] == DC_COUNTY_FIPS, :].copy()
    dc_results.loc[:, CommonFields.FIPS] = DC_STATE_FIPS
    dc_results.loc[:, CommonFields.AGGREGATE_LEVEL] = "state"

    results = pd.concat([results, dc_results])

    return remove_trailing_zeros(results)


class CDCTestingDataset(data_source.CanScraperBase):
    SOURCE_NAME = "CDCTesting"

    TRANSFORM_METHOD = transform

    EXPECTED_FIELDS = [
        CommonFields.TEST_POSITIVITY_7D,
    ]
