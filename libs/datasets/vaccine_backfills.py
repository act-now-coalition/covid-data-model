import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.pipeline import Region

MultiRegionDataset = timeseries.MultiRegionDataset


def derive_vaccine_pct(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Returns a new dataset with vaccination percentage metrics derived from their
    corresponding non-percentage fields where the percentage metric is missing or less fresh."""
    field_map = {
        CommonFields.VACCINATIONS_INITIATED: CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED: CommonFields.VACCINATIONS_COMPLETED_PCT,
    }

    ts_in_all = ds_in.timeseries_bucketed_wide_dates
    ts_in_all_variable_index = ts_in_all.index.get_level_values(PdFields.VARIABLE)
    ts_in_people = ts_in_all.loc[ts_in_all_variable_index.isin(field_map.keys())]
    ts_in_pcts = ts_in_all.loc[ts_in_all_variable_index.isin(field_map.values())]
    ts_in_without_pcts = ts_in_all.loc[~ts_in_all_variable_index.isin(field_map.values())]

    # TODO(tom): Preserve provenance and other tags from the original vaccination fields.
    # ds_in_wide_dates / dataset.static.loc[:, CommonFields.POPULATION] doesn't seem to align the
    # location_id correctly so be more explicit with `div`:
    derived_pct_df = (
        ts_in_people.div(
            ds_in.static.loc[:, CommonFields.POPULATION],
            level=CommonFields.LOCATION_ID,
            axis="index",
        )
        * 100.0
    )
    derived_pct_df = derived_pct_df.rename(index=field_map, level=PdFields.VARIABLE)

    def append_most_recent_date_index_level(df: pd.DataFrame) -> pd.DataFrame:
        """Appends most recent date with real (not NA) value as a new index level."""
        most_recent_date = df.apply(pd.Series.last_valid_index, axis=1)
        return df.assign(most_recent_date=most_recent_date).set_index(
            "most_recent_date", append=True
        )

    derived_pct_df = append_most_recent_date_index_level(derived_pct_df)
    ts_in_pcts = append_most_recent_date_index_level(ts_in_pcts)

    # Combine the input and derived percentage time series into one DataFrame, sort by most recent
    # date and drop duplicates except for the last/most recent.
    combined_pcts = (
        derived_pct_df.append(ts_in_pcts)
        .sort_index(level="most_recent_date")
        .droplevel("most_recent_date")
    )
    most_recent_pcts = combined_pcts.loc[~combined_pcts.index.duplicated(keep="last")]

    # Double check that there is no overlap between the time series in most_recent_pcts and
    # ts_in_without_pcts.
    assert ts_in_without_pcts.index.intersection(most_recent_pcts.index).empty

    return ds_in.replace_timeseries_wide_dates([ts_in_without_pcts, most_recent_pcts])


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
    computed_initiated = computed_initiated.dropna(axis=1, how="all")
    # Keep the computed initiated only where there is not already an existing time series.
    computed_initiated = computed_initiated.loc[
        ~computed_initiated.index.isin(existing_initiated.index)
    ]

    # Use concat to prepend the VARIABLE index level, then reorder the levels to match the dataset.
    computed_initiated = pd.concat(
        {CommonFields.VACCINATIONS_INITIATED: computed_initiated},
        names=[PdFields.VARIABLE] + list(computed_initiated.index.names),
    ).reorder_levels(timeseries_wide.index.names)

    return dataset.replace_timeseries_wide_dates(
        [timeseries_wide, computed_initiated]
    ).add_tag_to_subset(taglib.Derived(), computed_initiated.index)


STATE_LOCATION_ID = "state_location_id"


def add_state_location_id_index_level(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with the location_id of the state in a new level STATE_LOCATION_ID."""
    state_index = (
        df.index.get_level_values(CommonFields.LOCATION_ID)
        .map(lambda loc_id: Region.from_location_id(loc_id).get_state_region().location_id)
        .rename(STATE_LOCATION_ID)
    )
    return df.set_index(state_index, append=True)


def estimate_initiated_from_state_ratio(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    idx = pd.IndexSlice
    ts_states = ds_in.get_subset(
        AggregationLevel.STATE
    ).timeseries_not_bucketed_wide_dates.reorder_levels(
        [PdFields.VARIABLE, CommonFields.LOCATION_ID]
    )
    ts_state_level_ratios = (
        # ts.loc(axis=0)[field] returns a DataFrame with only one level in axis 0, an index
        # of location_ids.
        ts_states.loc(axis=0)[CommonFields.VACCINATIONS_INITIATED]
        / ts_states.loc(axis=0)[CommonFields.VACCINATIONS_COMPLETED]
    )
    ts_state_level_ratios = ts_state_level_ratios.rename_axis(
        index={CommonFields.LOCATION_ID: STATE_LOCATION_ID}
    )
    ts_counties = ds_in.get_subset(
        AggregationLevel.COUNTY
    ).timeseries_not_bucketed_wide_dates.reorder_levels(
        [PdFields.VARIABLE, CommonFields.LOCATION_ID]
    )
    ts_counties_initiated = ts_counties.loc(axis=0)[CommonFields.VACCINATIONS_INITIATED]
    ts_counties_completed = ts_counties.loc(axis=0)[CommonFields.VACCINATIONS_COMPLETED]
    counties_to_fix = ts_counties_completed.index.difference(ts_counties_initiated.index)
    ts_counties_completed_to_fix = add_state_location_id_index_level(
        ts_counties_completed.reindex(counties_to_fix)
    )
    ts_counties_initiated_est = ts_counties_completed_to_fix.mul(
        ts_state_level_ratios, level=STATE_LOCATION_ID
    ).droplevel(STATE_LOCATION_ID)
    ts_counties_initiated_est.index = pd.MultiIndex.from_product(
        (
            ts_counties_initiated_est.index,
            [CommonFields.VACCINATIONS_INITIATED],
            [DemographicBucket.ALL],
        ),
        names=[CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET],
    )

    return ds_in.replace_timeseries_wide_dates(
        [ds_in.timeseries_bucketed_wide_dates, ts_counties_initiated_est]
    )
