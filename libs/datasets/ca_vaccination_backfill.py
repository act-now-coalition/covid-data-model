import dataclasses
import pandas as pd
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets import AggregationLevel
from libs.pipeline import Region

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields


def derive_ca_county_vaccine_pct(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Derives vaccination metrics for CA counties based on State 1st vs 2nd dose reporting."""

    ca_county_dataset = ds_in.get_subset(aggregation_level=AggregationLevel.COUNTY, state="CA")
    ca_county_wide = ca_county_dataset.timeseries_wide_dates_no_buckets()
    fields_to_check = [
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
        CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED_PCT,
    ]
    # Assert that possible fields we want to estimate are all NA - if one of these is
    # not NA, likely do not need to estimate anymore and this methodology can be removed.
    assert ca_county_wide.loc[(slice(None), fields_to_check), :].isna().all().all()

    ca_state_wide = ds_in.get_regions_subset(
        [Region.from_state("CA")]
    ).timeseries_wide_dates_no_buckets()

    # Drop location index because not used to apply to county level data
    ca_state_wide = ca_state_wide.reset_index(CommonFields.LOCATION_ID, drop=True)

    ca_administered = ca_state_wide.loc[CommonFields.VACCINES_ADMINISTERED, :]

    initiated_ratio_of_administered = (
        ca_state_wide.loc[CommonFields.VACCINATIONS_INITIATED, :] / ca_administered
    )
    completed_ratio_of_administered = (
        ca_state_wide.loc[CommonFields.VACCINATIONS_COMPLETED, :] / ca_administered
    )

    county_administered = ca_county_wide.loc[(slice(None), [CommonFields.VACCINES_ADMINISTERED]), :]

    estimated_initiated = county_administered * initiated_ratio_of_administered
    estimated_completed = county_administered * completed_ratio_of_administered

    vaccines_initiated_pct = (
        estimated_initiated.div(
            ca_county_dataset.static.loc[:, CommonFields.POPULATION],
            level=CommonFields.LOCATION_ID,
            axis="index",
        )
        * 100
    )
    vaccines_initiated_pct = vaccines_initiated_pct.rename(
        index={CommonFields.VACCINES_ADMINISTERED: CommonFields.VACCINATIONS_INITIATED_PCT},
        level=PdFields.VARIABLE,
    )

    vaccines_completed_pct = (
        estimated_completed.div(
            ca_county_dataset.static.loc[:, CommonFields.POPULATION],
            level=CommonFields.LOCATION_ID,
            axis="index",
        )
        * 100
    )
    vaccines_completed_pct = vaccines_completed_pct.rename(
        index={CommonFields.VACCINES_ADMINISTERED: CommonFields.VACCINATIONS_COMPLETED_PCT},
        level=PdFields.VARIABLE,
    )

    all_wide = ds_in.timeseries_wide_dates_no_buckets()
    # Because we assert that existing dataset does not have CA county VACCINATIONS_COMPLETED_PCT
    # or VACCINATIONS_INITIATED_PCT we can safely combine the existing rows with new derived rows
    combined = pd.concat([vaccines_completed_pct, vaccines_initiated_pct, all_wide])

    return dataclasses.replace(
        ds_in,
        timeseries=MultiRegionDataset.from_timeseries_wide_dates_df(combined).timeseries,
        timeseries_bucketed=None,
    )
