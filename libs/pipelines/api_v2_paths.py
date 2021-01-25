from typing import Optional
import enum
import dataclasses
import pathlib
from libs.datasets import AggregationLevel
from api import can_api_v2_definition


class FileType(enum.Enum):
    CSV = 0
    JSON = 1

    @property
    def suffix(self):
        if self is FileType.CSV:
            return "csv"
        elif self is FileType.JSON:
            return "json"


@dataclasses.dataclass
class APIOutputPathBuilder:
    root: pathlib.Path
    level: AggregationLevel

    @property
    def region_key(self):
        if self.level is AggregationLevel.COUNTY:
            return "counties"
        if self.level is AggregationLevel.STATE:
            return "states"
        if self.level is AggregationLevel.CBSA:
            return "cbsas"
        if self.level is AggregationLevel.PLACE:
            return "places"

        raise ValueError("Unsupported aggregation level")

    @property
    def region_subdir(self) -> pathlib.Path:
        if self.level is AggregationLevel.COUNTY:
            return self.root / "county"
        if self.level is AggregationLevel.STATE:
            return self.root / "state"
        if self.level is AggregationLevel.CBSA:
            return self.root / "cbsa"
        if self.level is AggregationLevel.PLACE:
            return self.root / "place"

        raise ValueError("Unsupported aggregation level")

    def make_directories(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.region_subdir.mkdir(exist_ok=True)

    def bulk_timeseries(
        self,
        aggregate_timeseries_summary: can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
        file_type: FileType,
        state: Optional[str] = None,
    ) -> str:
        assert file_type is FileType.JSON

        if state:
            return self.region_subdir / f"{state}.timeseries.{file_type.suffix}"

        return self.root / f"{self.region_key}.timeseries.{file_type.suffix}"

    def bulk_summary(
        self,
        aggregate_summary: can_api_v2_definition.AggregateRegionSummary,
        file_type: FileType,
        state: Optional[str] = None,
    ) -> str:
        if state:
            return self.region_subdir / f"{state}.{file_type.suffix}"

        return self.root / f"{self.region_key}.{file_type.suffix}"

    def bulk_flattened_timeseries_data(self, file_type, state: Optional[str] = None):
        assert file_type is FileType.CSV

        if state:
            return self.region_subdir / f"{state}.timeseries.{file_type.suffix}"

        return self.root / f"{self.region_key}.timeseries.{file_type.suffix}"

    def single_summary(self, region_summary: can_api_v2_definition.RegionSummary, file_type):
        if self.level is AggregationLevel.STATE:
            return self.region_subdir / f"{region_summary.state}.{file_type.suffix}"
        if self.level in (AggregationLevel.COUNTY, AggregationLevel.CBSA, AggregationLevel.PLACE):
            return self.region_subdir / f"{region_summary.fips}.{file_type.suffix}"

        raise NotImplementedError("Level not supported")

    def single_timeseries(
        self, region_timeseries: can_api_v2_definition.RegionSummaryWithTimeseries, file_type
    ):

        if self.level is AggregationLevel.STATE:
            return self.region_subdir / f"{region_timeseries.state}.timeseries.{file_type.suffix}"
        if self.level in (AggregationLevel.COUNTY, AggregationLevel.CBSA, AggregationLevel.PLACE):
            return self.region_subdir / f"{region_timeseries.fips}.timeseries.{file_type.suffix}"

        raise NotImplementedError("Level not supported")
