from typing import Union
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset


class DatasetFilter(object):
    """Defines a filter to apply to a dataset."""

    def __init__(self, aggregation_level: AggregationLevel = None, **filters):
        """
        Args:
            aggregation_level: AggregationLevel of dataset
            filters: Filters to apply to dataset.  Must match arguments of the respective
                `get_subset` function.
        """
        self.aggregation_level = aggregation_level
        self.filters = filters

    def apply(self, dataset: Union[TimeseriesDataset, LatestValuesDataset]):
        """Applies filter to dataset by returning subset that matches filter arguments."""
        return dataset.get_subset(self.aggregation_level, **self.filters)
