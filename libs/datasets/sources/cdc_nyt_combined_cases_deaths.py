import dataclasses
from functools import lru_cache

import pandas as pd
from libs.datasets import data_source
from datapublic.common_fields import CommonFields, PdFields
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.new_cases_and_deaths import add_new_cases
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.timeseries import MultiRegionDataset


NYT_CUTOFF_DATE = "2023-01-15"


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

        # CDC state-level data only provides weekly datapoints, so we fill forward to get "daily"
        # granularity. This conforms the data to the format of the NYT data, which the parts
        # of the pipeline expects.
        def _forward_fill_field(
            dataset: MultiRegionDataset, field: CommonFields
        ) -> MultiRegionDataset:
            # timeseries_bucketed_wide_dates preserves all dates even if they are NaN
            # so we can use it to fill forward.
            ts_wide: pd.DataFrame = dataset.timeseries_bucketed_wide_dates
            field_ts = ts_wide.xs(field, level=PdFields.VARIABLE, drop_level=False)
            filled_ts = field_ts.ffill(limit=6, axis=1)
            filled_ds = MultiRegionDataset.from_timeseries_wide_dates_df(filled_ts, bucketed=True)
            return dataset.drop_column_if_present(field).join_columns(filled_ds)

        with_filled_cases_ds = _forward_fill_field(dataset, CommonFields.CASES)
        return _forward_fill_field(with_filled_cases_ds, CommonFields.DEATHS)


class CdcNytCombinedCasesDeaths(data_source.DataSource):
    """Data source combining the CDC's historical and as-originally-posted datasets."""

    SOURCE_TYPE = "CdcNytCombinedCasesDeaths"
    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        """Use historical data when possible, use as-originally-posted for data that the historical dataset does not have.
        """

        nyt_ds = NYTimesDataset.make_dataset()
        cdc_ds = CDCCasesDeaths.make_dataset()

        nyt_ts: pd.DataFrame = nyt_ds.timeseries.query(f"{CommonFields.DATE} < '{NYT_CUTOFF_DATE}'")
        cdc_ts: pd.DataFrame = cdc_ds.timeseries.query(
            f"{CommonFields.DATE} >= '{NYT_CUTOFF_DATE}'"
        )

        # TODO: fix the tags
        combined_ts = nyt_ts.combine_first(cdc_ts)
        merged_ds: MultiRegionDataset = dataclasses.replace(
            nyt_ds, timeseries=combined_ts, timeseries_bucketed=None,
        )
        merged_ds.to_compressed_pickle(REPO_ROOT / "cdc_nyt_combined_cases_deaths.pkl.gz")
        return add_new_cases(merged_ds)
