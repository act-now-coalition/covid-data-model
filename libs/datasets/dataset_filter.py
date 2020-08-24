from typing import List

from pydantic.dataclasses import dataclass

from libs.datasets.dataset_base import DatasetBase


@dataclass
class DatasetFilter(object):
    """Defines a filter to apply to a dataset."""

    country: str
    states: List[str]

    def apply(self, dataset: DatasetBase) -> DatasetBase:
        """Applies filter to dataset by returning subset that matches filter arguments."""
        aggregation_level = None
        return dataset.get_subset(
            aggregation_level=aggregation_level, country=self.country, states=self.states
        )
