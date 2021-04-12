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
    ds_in_wide_dates = ds_in.timeseries_not_bucketed_wide_dates
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


def _xs_or_empty(df: pd.DataFrame, key, *, level) -> pd.DataFrame:
    """Similar to df.xs(key, level=level, drop_level=True) but returns an empty DataFrame instead of
    raising KeyError when key is not found."""
    index_mask = df.index.get_level_values(level) == key
    rows = df.loc(axis=0)[index_mask]
    return rows.droplevel(level)


def backfill_vaccination_initiated(dataset: MultiRegionDataset) -> MultiRegionDataset:
    """Backfills vaccination initiated data from total doses administered and total completed.

    Args:
        dataset: Input dataset.

    Returns: New dataset with backfilled data.
    """
    timeseries_wide = dataset.timeseries_bucketed_wide_dates.dropna(axis=1, how="all")

    administered = _xs_or_empty(
        timeseries_wide, CommonFields.VACCINES_ADMINISTERED, level=PdFields.VARIABLE
    )
    completed = _xs_or_empty(
        timeseries_wide, CommonFields.VACCINATIONS_COMPLETED, level=PdFields.VARIABLE
    )
    existing_initiated = _xs_or_empty(
        timeseries_wide, CommonFields.VACCINATIONS_INITIATED, level=PdFields.VARIABLE
    )

    # Compute and keep only time series with at least one real value
    computed_initiated = administered - completed

    # Use concat to prepend the VARIABLE index level, then reorder the levels to match the dataset.
    computed_initiated = pd.concat(
        {CommonFields.VACCINATIONS_INITIATED: computed_initiated},
        names=[PdFields.VARIABLE] + list(computed_initiated.index.names),
    ).reorder_levels(timeseries_wide.index.names)

    timeseries_wide_combined = pd.concat([timeseries_wide, computed_initiated])

    # https://stackoverflow.com/a/34297689
    timeseries_wide_deduped = timeseries_wide_combined.loc[
        ~timeseries_wide_combined.index.duplicated(keep="first")
    ]

    computed_initiated = computed_initiated.dropna(axis=1, how="all")
    computed_initiated = computed_initiated.loc[
        ~computed_initiated.droplevel(PdFields.VARIABLE).index.isin(existing_initiated.index)
    ]
    timeseries_wide_deduped_2 = pd.concat([timeseries_wide, computed_initiated])

    assert timeseries_wide_deduped_2.equals(timeseries_wide_deduped)

    timeseries_wide_vars = timeseries_wide_deduped.stack().unstack(PdFields.VARIABLE)

    return dataclasses.replace(dataset, timeseries_bucketed=timeseries_wide_vars)
