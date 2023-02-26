from functools import lru_cache

import pandas as pd
from libs.datasets import data_source, taglib
from datapublic.common_fields import CommonFields, PdFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.timeseries import MultiRegionDataset


NYT_CUTOFF_DATE = "2023-03-01"


class CDCCasesDeaths(data_source.CanScraperBase):
    SOURCE_TYPE = "CDC"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.DEATHS,
        ),
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        """Returns a dataset with cumulative cases and deaths from CDC."""
        dataset = super().make_dataset()

        # CDC state-level data only provides weekly datapoints, so we
        # add in the missing dates and fill forward to get "daily"
        # granularity. This conforms the data to the format of the NYT
        # data, which parts of the pipeline expect.
        def _forward_fill_field(
            dataset: MultiRegionDataset, field: CommonFields
        ) -> MultiRegionDataset:
            # timeseries_bucketed_wide_dates preserves all dates even if they are NaN
            # so we can use it to fill forward.
            ts_filled = dataset.timeseries_bucketed_wide_dates.xs(
                field, level=PdFields.VARIABLE, drop_level=False
            ).ffill(axis=1)
            filled_ds = MultiRegionDataset.from_timeseries_wide_dates_df(ts_filled, bucketed=True)
            return dataset.drop_column_if_present(field).join_columns(filled_ds)

        with_ffilled_cases_ds = _forward_fill_field(dataset, CommonFields.CASES)
        return _forward_fill_field(with_ffilled_cases_ds, CommonFields.DEATHS)


class CdcNytCombinedCasesDeaths(data_source.DataSource):
    """Data source combining the CDC's historical and as-originally-posted datasets."""

    SOURCE_TYPE = "CdcNytCombinedCasesDeaths"
    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        """Combines the CDC and NYT datasets into a single dataset."""
        nyt_ds = NYTimesDataset.make_dataset()
        cdc_ds = CDCCasesDeaths.make_dataset()

        nyt_ts: pd.DataFrame = nyt_ds.timeseries.query(f"{CommonFields.DATE} < '{NYT_CUTOFF_DATE}'")
        cdc_ts: pd.DataFrame = cdc_ds.timeseries.query(
            f"{CommonFields.DATE} >= '{NYT_CUTOFF_DATE}'"
        )

        # Manually creating tags for simplicity's sake
        tag = taglib.Source(
            type="CdcNyt",
            url=[
                "https://github.com/nytimes/covid-19-data",
                "https://covid.cdc.gov/covid-data-tracker/#datatracker-home",
            ],
            name=f"NYT data before {NYT_CUTOFF_DATE}, CDC data after {NYT_CUTOFF_DATE}",
        )

        combined_ts = nyt_ts.combine_first(cdc_ts)
        return MultiRegionDataset(timeseries=combined_ts).add_tag_all_bucket(tag)
