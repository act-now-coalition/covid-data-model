import dataclasses

import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import timeseries


MultiRegionDataset = timeseries.MultiRegionDataset


def derive_vaccine_pct(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Returns a new dataset containing everything all the input and vaccination percentage
    metrics derived from the corresponding non-percentage field where not already set."""
    field_map = {
        CommonFields.VACCINATIONS_INITIATED: CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED: CommonFields.VACCINATIONS_COMPLETED_PCT,
    }
    ds_in_wide_dates = ds_in.timeseries_wide_dates_no_buckets
    ds_in_wide_dates = ds_in_wide_dates.loc[
        ds_in_wide_dates.index.get_level_values(PdFields.VARIABLE).isin(field_map.keys())
    ]
    # TODO(tom): Preserve provenance and other tags from the original vaccination fields.
    # ds_in_wide_dates / dataset.static.loc[:, CommonFields.POPULATION] doesn't seem to align the
    # location_id correctly so be more explicit with `div`:
    derived_pct_df = (
        ds_in_wide_dates.div(
            ds_in.static.loc[:, CommonFields.POPULATION],
            level=CommonFields.LOCATION_ID,
            axis="index",
        )
        * 100.0
    )
    derived_pct_df = derived_pct_df.rename(index=field_map, level=PdFields.VARIABLE)
    # Make a dataset containing only the derived percentage metrics.
    ds_derived_pct = MultiRegionDataset.from_timeseries_wide_dates_df(derived_pct_df)

    # Combine the derived percentage with any percentages in ds_in to make a dataset containing
    # only the percentage metrics.
    # Possible optimization: Replace the combine+drop+join with a single function that copies from
    # ds_derived_pct only where a timeseries is all NaN in the original dataset.
    ds_list = [ds_in, ds_derived_pct]
    ds_all_pct = timeseries.combined_datasets(
        {field_pct: ds_list for field_pct in field_map.values()}, {}
    )

    dataset_without_pct = ds_in.drop_columns_if_present(list(field_map.values()))
    return dataset_without_pct.join_columns(ds_all_pct)


def backfill_vaccination_initiated(dataset: MultiRegionDataset) -> MultiRegionDataset:
    """Backfills vaccination initiated data from total doses administered and total completed.

    Args:
        dataset: Input dataset.

    Returns: New dataset with backfilled data.
    """
    fields = [
        CommonFields.VACCINES_ADMINISTERED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
    ]
    df = dataset.timeseries_wide_dates_no_buckets.loc[(slice(None), fields), :]
    df_var_first = df.reorder_levels([PdFields.VARIABLE, CommonFields.LOCATION_ID])

    administered = df_var_first.loc[CommonFields.VACCINES_ADMINISTERED]
    inititiated = df_var_first.loc[[CommonFields.VACCINATIONS_INITIATED]]
    completed = df_var_first.loc[[CommonFields.VACCINATIONS_COMPLETED]]

    computed_initiated = administered - completed

    # Rename index value to be initiated
    computed_initiated = computed_initiated.rename(
        index={CommonFields.VACCINATIONS_COMPLETED: CommonFields.VACCINATIONS_INITIATED}
    )

    locations = df.index.get_level_values(CommonFields.LOCATION_ID).unique()
    locations_with_initiated = (
        inititiated.notna()
        .any(axis="columns")
        .index.get_level_values(CommonFields.LOCATION_ID)
        .unique()
    )
    locations_without_initiated = locations.difference(locations_with_initiated)

    combined_initiated_df = pd.concat(
        [
            inititiated.loc[
                (slice(CommonFields.VACCINATIONS_INITIATED), locations_with_initiated), :
            ],
            computed_initiated.loc[
                (slice(CommonFields.VACCINATIONS_INITIATED), locations_without_initiated), :
            ],
        ]
    )

    combined_initiated_df = combined_initiated_df.reorder_levels(
        [CommonFields.LOCATION_ID, PdFields.VARIABLE]
    )
    initiated_dataset = MultiRegionDataset.from_timeseries_wide_dates_df(combined_initiated_df)

    # locations that are na for all vaccinations initiated
    timeseries_copy = dataset.timeseries.copy()
    timeseries_copy.loc[:, CommonFields.VACCINATIONS_INITIATED] = initiated_dataset.timeseries.loc[
        :, CommonFields.VACCINATIONS_INITIATED
    ]

    return dataclasses.replace(dataset, timeseries=timeseries_copy, timeseries_bucketed=None)
