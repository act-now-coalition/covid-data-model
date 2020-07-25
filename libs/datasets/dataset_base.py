from typing import List, Union, TextIO, Mapping, Iterable, Optional
import pathlib

import structlog

from covidactnow.datapublic import common_df
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields, COMMON_FIELDS_TIMESERIES_KEYS
from libs.datasets.dataset_utils import AggregationLevel


class DatasetBase(object):

    INDEX_FIELDS: List[str] = []

    COMMON_INDEX_FIELDS: List[str] = []

    def __init__(self, data: pd.DataFrame, provenance: Optional[pd.Series] = None):
        self.data = data
        self.provenance = provenance

    def get_subset(self, aggregation_level: AggregationLevel, **filters) -> "DatasetBase":
        """Returns a subset of the existing dataset."""
        raise NotImplementedError("Subsclass must implement")

    def yield_records(self) -> Iterable[dict]:
        # It'd be faster to use self.data.itertuples or find a way to avoid yield_records, but that
        # needs larger changes in code calling this.
        for idx, row in self.data.iterrows():
            yield row.where(pd.notnull(row), None).to_dict()

    @classmethod
    def build_from_data_source(cls, source) -> "DatasetBase":
        """Builds an instance of the dataset from a data source."""
        raise NotImplementedError("Subsclass must implement")

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
