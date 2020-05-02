from typing import List
import functools
import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.location_metadata import MetadataDataset
from libs.datasets.sources.jhu_dataset import JHUDataset
from libs.datasets.sources.nha_hospitalization import NevadaHospitalAssociationData
from libs.datasets.sources.fips_population import FIPSPopulation
from libs.datasets.sources.covid_care_map import CovidCareMapBeds
from libs.datasets.sources.cds_dataset import CDSDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource


@functools.lru_cache(None)
def aggregate_timeseries():
    return AggregateActualTimeseries.initialize()


@functools.lru_cache(None)
def combined_latest():
    return CombinedLatestValues.initialize()


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
        Fields.CURRENT_ICU: [CovidTrackingDataSource, NevadaHospitalAssociationData],
        Fields.CURRENT_HOSPITALIZED: [CovidTrackingDataSource, NevadaHospitalAssociationData],
        Fields.POSITIVE_TESTS: [CDSDataset, CovidTrackingDataSource],
        Fields.NEGATIVE_TESTS: [CDSDataset, CovidTrackingDataSource]
    }

    @classmethod
    def initialize(cls) -> MetadataDataset:
        # Some initial data frame
        data = pd.DataFrame({})
        index_fields = [
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        loaded_data_sources = {}

        for key, data_source_classes in cls.DATA_SOURCE_MAP.items():

            if not isinstance(data_source_classes, list):
                data_source_classes = [data_source_classes]

            for data_source_cls in data_source_classes:
                # only load data source once
                if data_source_cls not in loaded_data_sources:
                    loaded_data_sources[data_source_cls] = data_source_cls.local().metadata()

        for field, data_source_classes in cls.DATA_SOURCE_MAP.items():
            columns_to_fill = [field]
            if not isinstance(data_source_classes, list):
                data_source_classes = [data_source_classes]

            for data_source_cls in data_source_classes:
                data_source = loaded_data_sources[data_source_cls]
                data = dataset_utils.fill_fields_with_data_source(
                    data, data_source.data, index_fields, columns_to_fill
                )

        return MetadataDataset(data)


class AggregateActualTimeseries(object):

    Fields = TimeseriesDataset.Fields

    # Data source map takes field from data source specified.
    # If data sources is a list, will start with left most data source and overwrite with
    # values from other datasets.
    # so: Fields.CURRENT_ICU: [CovidTrackingDataSource, NevadaHospitalAssociationData]
    # will load all icu data from covid tracking and then overwrite matching fiels in NHA data.
    DATA_SOURCE_MAP = {
        Fields.CASES: JHUDataset,
        Fields.DEATHS: JHUDataset,
        Fields.CURRENT_HOSPITALIZED: [CovidTrackingDataSource, NevadaHospitalAssociationData],
        Fields.CURRENT_VENTILATED: [CovidTrackingDataSource, NevadaHospitalAssociationData],
        Fields.CUMULATIVE_ICU: CovidTrackingDataSource,
        Fields.CUMULATIVE_HOSPITALIZED: CovidTrackingDataSource,
        Fields.CURRENT_ICU: [CovidTrackingDataSource, NevadaHospitalAssociationData],
        Fields.POSITIVE_TESTS: [CDSDataset, CovidTrackingDataSource],
        Fields.NEGATIVE_TESTS: [CDSDataset, CovidTrackingDataSource]
    }


    @classmethod
    def initialize(cls) -> MetadataDataset:
        # Some initial data frame
        data = pd.DataFrame({})
        index_fields = [
            cls.Fields.DATE,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
            cls.Fields.FIPS,
        ]
        loaded_data_sources = {}

        for key, data_source_classes in cls.DATA_SOURCE_MAP.items():

            if not isinstance(data_source_classes, list):
                data_source_classes = [data_source_classes]

            for data_source_cls in data_source_classes:
                # only load data source once
                if data_source_cls not in loaded_data_sources:
                    loaded_data_sources[data_source_cls] = data_source_cls.local().timeseries()

        for field, data_source_classes in cls.DATA_SOURCE_MAP.items():
            columns_to_fill = [field]
            if not isinstance(data_source_classes, list):
                data_source_classes = [data_source_classes]

            for data_source_cls in data_source_classes:
                data_source = loaded_data_sources[data_source_cls]
                data = dataset_utils.fill_fields_with_data_source(
                    data, data_source.data, index_fields, columns_to_fill
                )

        return TimeseriesDataset(data)
