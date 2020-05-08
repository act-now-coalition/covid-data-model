from typing import List
import pandas as pd
from libs.datasets.dataset_utils import AggregationLevel


class DatasetBase(object):

    # Class of Fields in this dataset
    Fields = None

    INDEX_FIELDS: List[str] = []

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_subset(
        self, aggregation_level: AggregationLevel, **filters
    ) -> "DatasetBase":
        """Returns a subset of the existing dataset."""
        raise NotImplementedError("Subsclass must implement")

    @classmethod
    def build_from_data_source(cls, source) -> "DatasetBase":
        """Builds an instance of the dataset from a data source."""
        raise NotImplementedError("Subsclass must implement")
