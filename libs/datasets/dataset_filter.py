from typing import Union, List

from pydantic.dataclasses import dataclass

from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset


@dataclass
class DatasetFilter(object):
    """Defines a filter to apply to a dataset."""

    country: str
    states: List[str]

    def apply(self, dataset: Union[TimeseriesDataset, LatestValuesDataset]):
        """Applies filter to dataset by returning subset that matches filter arguments."""
        return dataset.get_subset(aggregation_level=None, country=self.country, states=self.states)
