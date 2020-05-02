from typing import Type, List

import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.location_metadata import MetadataDataset
from libs.datasets.sources.jhu_dataset import JHUDataset
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource


def _add_columns_to_data(
    existing_df: pd.DataFrame,
    data_source: TimeseriesDataset,
    index_fields: List[str],
    columns_to_fill: List[str],
):
    """Adds columns from data source to aggregate data

    Returns: Updated dataframe with
    """
    # data source-specific columns to be use in the aggregate result.
    new_data = data_source.set_index(index_fields)

    # If no data exists, return all rows from new data with just the requested columns.
    if not len(existing_df):
        for column in columns_to_fill:
            if column not in new_data.columns:
                new_data[column] = None
        return new_data[columns_to_fill].reset_index()
    existing_df = existing_df.set_index(index_fields)

    # Sort indices so that we have chunks of equal length in the
    # correct order so that we can splice in values from nevada data.
    existing_df = existing_df.sort_index()
    new_data = new_data.sort_index()
    # Returns a series
    # -----------
    # returns a boolean list of existing df rows that have an index match in new_data
    existing_df_in_new_data = existing_df.index.isin(new_data.index)
    new_data_in_existing_df = new_data.index.isin(existing_df.index)

    if not sum(existing_df_in_new_data) == sum(new_data_in_existing_df):
        raise ValueError("Number of rows should be the for data to replace")

    # Get columns that this data source should represent.
    for column in columns_to_fill:
        existing_df[column] = None

    # Fill in columns that match the index
    existing_df.loc[existing_df_in_new_data, columns_to_fill] = new_data.loc[
        new_data_in_existing_df, columns_to_fill
    ]
    # Combine updated data with rows not present in covid tracking data.
    # print("HI")
    # print(new_data[~new_data_in_existing_df][columns_to_fill].head())

    data = pd.concat(
        [
            existing_df.reset_index(),
            # rows that doesn't have a match with
            new_data[~new_data_in_existing_df][columns_to_fill].reset_index(),
        ]
    )
    # print(data[data.state == "MA"].head(100))
    return data


class CombinedLatestValues(object):
    Fields = MetadataDataset.Fields

    DATA_SOURCE_MAP = {
        Fields.STAFFED_BEDS: CovidCareMapBeds,
        Fields.LICENSED_BEDS: CovidCareMapBeds,
        Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE: CovidCareMapBeds,
        Fields.ICU_TYPICAL_OCCUPANCY_RATE: CovidCareMapBeds,
        Fields.ICU_BEDS: CovidCareMapBeds,
        Fields.POPULATION: FIPSPopulation,
        Fields.DEATHS: JHUDataset,
        Fields.CASES: JHUDataset,
        Fields.CURRENT_ICU: CovidTrackingDataSource,
        Fields.CURRENT_HOSPITALIZED: CovidTrackingDataSource,
    }

    @classmethod
    def initialize(cls) -> MetadataDataset:
        # Some initial data frame
        data = pd.DataFrame({})
        classes = set(cls.DATA_SOURCE_MAP.values())
        index_fields = [
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        state_index_fields = [
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
        ]
        loaded_data_with_cls = [
            (source_cls.local().metadata(), source_cls) for source_cls in classes
        ]
        for data_source, source_cls in loaded_data_with_cls:
            columns_to_fill = [
                key for key, value in cls.DATA_SOURCE_MAP.items() if value == source_cls
            ]
            if not len(data):
                state_data = pd.DataFrame({})
                county_data = pd.DataFrame({})
            else:
                state_data = data[data.aggregate_level == 'state']
                county_data = data[data.aggregate_level == 'county']
            state_data = _add_columns_to_data(
                state_data, data_source.state_data, state_index_fields, columns_to_fill
            )
            county_data = _add_columns_to_data(
                county_data, data_source.county_data, index_fields, columns_to_fill
            )
            data = pd.concat([state_data, county_data])

        return MetadataDataset(data)


class AggregateActualTimeseries(object):

    Fields = TimeseriesDataset.Fields

    DATA_SOURCE_MAP = {
        Fields.CASES: JHUDataset,
        Fields.DEATHS: JHUDataset,
        Fields.CURRENT_HOSPITALIZED: CovidTrackingDataSource,
        Fields.CUMULATIVE_ICU: CovidTrackingDataSource,
        Fields.CURRENT_ICU: CovidTrackingDataSource,
    }

    @classmethod
    def initialize(cls) -> TimeseriesDataset:
        # Some initial data frame

        data = pd.DataFrame({})
        classes = set(cls.DATA_SOURCE_MAP.values())
        index_fields = [
            cls.Fields.DATE,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        loaded_data_with_cls = [
            (source_cls.local().timeseries(), source_cls) for source_cls in classes
        ]
        for data_source, source_cls in loaded_data_with_cls:
            columns_to_fill = [
                key for key, value in cls.DATA_SOURCE_MAP.items() if value == source_cls
            ]

            data = _add_columns_to_data(
                data, data_source.data, index_fields, columns_to_fill
            )

        return TimeseriesDataset(data)
