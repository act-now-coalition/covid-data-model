from typing import List

import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.location_metadata import MetadataDataset
from libs.datasets.sources.jhu_dataset import JHUDataset
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource


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
        loaded_data_with_cls = [
            (source_cls.local().metadata(), source_cls) for source_cls in classes
        ]
        for data_source, source_cls in loaded_data_with_cls:
            columns_to_fill = [
                key for key, value in cls.DATA_SOURCE_MAP.items() if value == source_cls
            ]
            data = dataset_utils.fill_fields_with_data_source(
                data, data_source.data, index_fields, columns_to_fill
            )

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
            (source_cls.local().timeseries(), source_cls)
            for source_cls in classes
        ]
        for data_source, source_cls in loaded_data_with_cls:
            columns_to_fill = [
                key for key, value in cls.DATA_SOURCE_MAP.items() if value == source_cls
            ]

            data = dataset_utils.fill_fields_with_data_source(
                data, data_source.data, index_fields, columns_to_fill
            )

        return TimeseriesDataset(data)
