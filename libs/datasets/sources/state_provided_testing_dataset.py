import dataclasses
from libs.datasets import data_source
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.timeseries import MultiRegionDataset


class StateProvidedTestingDataset(data_source.CanScraperBase):
    """Data source connecting to state health department sourced testing data."""

    SOURCE_TYPE = "StateProvidedTesting"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="cumulative",
            provider="state",
            unit="specimens",
            common_field=CommonFields.POSITIVE_TESTS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_total",
            measurement="cumulative",
            provider="state",
            unit="specimens",
            common_field=CommonFields.TOTAL_TESTS,
        ),
    ]

    def make_dataset(self) -> MultiRegionDataset:
        dataset = super().make_dataset()
        ts = dataset.timeseries_bucketed
        # Calculate 7-day rolling sum to match CDC test positivity convention.
        # See "Test Percent Positivity Metric" at
        # https://data.cdc.gov/Public-Health-Surveillance/Weekly-COVID-19-County-Level-of-Community-Transmis/dt66-w6m6/
        ts[CommonFields.TOTAL_TESTS_7D] = ts[CommonFields.TOTAL_TESTS].rolling(window=7).sum()
        ts[CommonFields.POSITIVE_TESTS_7D] = ts[CommonFields.POSITIVE_TESTS].rolling(window=7).sum()
        ts = ts.drop(columns=[CommonFields.TOTAL_TESTS, CommonFields.POSITIVE_TESTS])

        return dataclasses.replace(dataset, timeseries_bucketed=ts)
