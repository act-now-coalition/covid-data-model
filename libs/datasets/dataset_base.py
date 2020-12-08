from typing import List, Union, TextIO, Iterable, Optional
import pathlib

import structlog

from covidactnow.datapublic import common_df
import pandas as pd

from libs.datasets.dataset_utils import AggregationLevel, DatasetType


class SaveableDatasetInterface:
    def to_csv(self, path: pathlib.Path):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        raise NotImplementedError("Subsclass must implement")

    @property
    def dataset_type(self) -> DatasetType:
        raise NotImplementedError("Subsclass must implement")


class DatasetBase(SaveableDatasetInterface):

    INDEX_FIELDS: List[str] = []

    COMMON_INDEX_FIELDS: List[str] = []

    def __init__(self, data: pd.DataFrame, provenance: Optional[pd.Series] = None):
        self.data = data
        self.provenance = provenance

    def get_subset(self, aggregation_level: AggregationLevel, **filters) -> "DatasetBase":
        """Returns a subset of the existing dataset."""
        raise NotImplementedError("Subsclass must implement")

    def yield_records(self) -> Iterable[dict]:
        # TODO(tom): This function is only called from tests. Replace the calls and remove it.
        for idx, row in self.data.iterrows():
            yield row.where(pd.notnull(row), None).to_dict()

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        raise NotImplementedError()

    def to_csv(self, path: pathlib.Path):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        common_df.write_csv(self.data, path, structlog.get_logger(), self.COMMON_INDEX_FIELDS)
        if self.provenance is not None:
            provenance_path = str(path).replace(".csv", "-provenance.csv")
            self.provenance.sort_index().to_csv(provenance_path)
