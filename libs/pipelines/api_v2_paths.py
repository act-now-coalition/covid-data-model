import enum
import pathlib
from libs.datasets import AggregationLevel


class FileType(enum.Enum):
    CSV = 0
    JSON = 1

    @property
    def suffix(self):
        if self is FileType.CSV:
            return "csv"
        elif self is FileType.JSON:
            return "json"


def _region_key(level):
    if level is AggregationLevel.COUNTY:
        return "counties"
    if level is AggregationLevel.STATE:
        return "states"


class APIOutputPathBuilder:
    def __init__(self, root: pathlib.Path):
        self.root = root

    def make_directories(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "counties").mkdir(exist_ok=True)
        (self.root / "states").mkdir(exist_ok=True)

    def bulk_timeseries(self, aggregate_timeseries_summary, file_type: FileType) -> str:
        assert file_type is FileType.JSON
        key = _region_key(aggregate_timeseries_summary.level)
        return self.root / f"{key}.timeseries.{file_type.suffix}"

    def bulk_summary(self, aggregate_summary, file_type: FileType) -> str:
        key = _region_key(aggregate_summary.level)
        return self.root / f"{key}.{file_type.suffix}"

    def bulk_prediction_data(self, flattened_timeseries, file_type):
        assert file_type is FileType.CSV
        key = _region_key(flattened_timeseries.level)
        return self.root / f"{key}.{file_type.suffix}"

    def single_summary(self, region_summary, file_type):
        key = _region_key(region_summary.level)
        return self.root / key / f"{region_summary.fips}.{file_type.suffix}"

    def single_timeseries(self, region_timeseries, file_type):
        key = _region_key(region_timeseries.level)
        return self.root / key / f"{region_timeseries.fips}.timeseries.{file_type.suffix}"
