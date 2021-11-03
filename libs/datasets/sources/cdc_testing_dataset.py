import dataclasses
from functools import lru_cache

from libs.datasets import AggregationLevel
from libs.datasets import data_source
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.timeseries import MultiRegionDataset
from libs.pipeline import Region

DC_COUNTY_LOCATION_ID = Region.from_fips("11001").location_id
DC_STATE_LOCATION_ID = Region.from_state("DC").location_id


def _remove_trailing_zeros(series: pd.Series) -> pd.Series:

    series = pd.Series(series.values.copy(), index=series.index.get_level_values(CommonFields.DATE))

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
    data = data.sort_index()
    test_pos = data.groupby(CommonFields.LOCATION_ID)[CommonFields.TEST_POSITIVITY_7D].apply(
        _remove_trailing_zeros
    )
    data[CommonFields.TEST_POSITIVITY_7D] = test_pos
    return data


class CDCTestingDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "CDCTesting"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="rolling_average_7_day",
            provider="cdc",
            unit="percentage",
            common_field=CommonFields.TEST_POSITIVITY_7D,
        ),
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


class CDCHistoricalTestingDataset(data_source.CanScraperBase):
    """Data source connecting to the official CDC test positivity dataset.
    
    We prioritize this source over the old CDCTesting source, which scraped data from the CDC's
    internal API. 
    """

    SOURCE_TYPE = "CDCHistoricalTesting"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="rolling_average_7_day",
            provider="cdc2",
            unit="percentage",
            common_field=CommonFields.TEST_POSITIVITY_7D,
        ),
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return modify_dataset(super().make_dataset())


def modify_dataset(ds: MultiRegionDataset) -> MultiRegionDataset:
    ts_copy = ds.timeseries.copy()
    # Test positivity should be a ratio
    ts_copy.loc[:, CommonFields.TEST_POSITIVITY_7D] = (
        ts_copy.loc[:, CommonFields.TEST_POSITIVITY_7D] / 100.0
    )

    levels = set(
        Region.from_location_id(l).level
        for l in ds.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
    )
    # Should only be picking up county all_df for now.  May need additional logic if states
    # are included as well
    assert levels == {AggregationLevel.COUNTY}

    # Duplicating DC County results as state results because of a downstream
    # use of how dc state data is used to override DC county data.
    dc_results = ts_copy.xs(
        DC_COUNTY_LOCATION_ID, axis=0, level=CommonFields.LOCATION_ID, drop_level=False
    )
    dc_results = dc_results.rename(
        index={DC_COUNTY_LOCATION_ID: DC_STATE_LOCATION_ID}, level=CommonFields.LOCATION_ID
    )

    ts_copy = ts_copy.append(dc_results, verify_integrity=True).sort_index()

    return dataclasses.replace(
        ds, timeseries=remove_trailing_zeros(ts_copy), timeseries_bucketed=None
    )
