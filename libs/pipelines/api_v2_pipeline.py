from typing import Iterator, List, Optional
from dataclasses import dataclass
import functools
import multiprocessing
import pathlib

import pydantic
import structlog

from libs.pipelines.api_v2_paths import APIOutputPathBuilder
from libs.pipelines.api_v2_paths import FileType
from api.can_api_v2_definition import AggregateRegionSummary
from api.can_api_v2_definition import AggregateRegionSummaryWithTimeseries
from api.can_api_v2_definition import Metrics
from api.can_api_v2_definition import MetricsTimeseriesRow
from api.can_api_v2_definition import RegionSummaryWithTimeseries
from libs import dataset_deployer
from libs import top_level_metrics
from libs import pipeline
from libs.datasets import CommonFields
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention
from libs.functions import generate_api_v2
from libs.functions import get_can_projection
from libs.datasets import combined_datasets

logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"

INTERVENTION = Intervention.OBSERVED_INTERVENTION


@dataclass(frozen=True)
class RegionalInput:
    region: pipeline.Region

    model_output: Optional[CANPyseirLocationOutput]

    _combined_data: combined_datasets.RegionalData

    @property
    def fips(self) -> str:
        return self.region.fips

    @property
    def latest(self):
        return self._combined_data.latest

    @property
    def timeseries(self):
        return self._combined_data.timeseries

    @staticmethod
    def from_region_and_model_output(
        region: pipeline.Region, model_output_dir: pathlib.Path
    ) -> "RegionalInput":
        combined_data = combined_datasets.RegionalData.from_region(region)

        model_output = CANPyseirLocationOutput.load_from_model_output_if_exists(
            region.fips, INTERVENTION, model_output_dir
        )
        return RegionalInput(region=region, model_output=model_output, _combined_data=combined_data)


def run_on_regions(
    regional_inputs: List[RegionalInput],
    pool: multiprocessing.Pool = None,
    sort_func=None,
    limit=None,
) -> Iterator[RegionSummaryWithTimeseries]:
    # Setting maxtasksperchild to one ensures that we minimize memory usage over time by creating
    # a new child for every task. Addresses OOMs we saw on highly parallel build machine.

    pool = pool or multiprocessing.Pool(maxtasksperchild=1)
    results = map(build_timeseries_for_region, regional_inputs)
    all_timeseries = []

    for region_timeseries in results:
        if not region_timeseries:
            continue

        all_timeseries.append(region_timeseries)

    if sort_func:
        all_timeseries.sort(key=sort_func)

    if limit:
        all_timeseries = all_timeseries[:limit]

    return all_timeseries


def generate_metrics_and_latest(
    timeseries: TimeseriesDataset, latest: dict, model_output: Optional[CANPyseirLocationOutput],
) -> [List[MetricsTimeseriesRow], Optional[Metrics]]:
    """
    For a FIPS, generate a MetricsTimeseriesRow per day and return the latest.

    Args:
        fips: FIPS to run on.

    Returns:
        Tuple of MetricsTimeseriesRows for all days and the latest.
    """
    metrics_results, latest = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, latest, model_output
    )
    metrics_timeseries = metrics_results.to_dict(orient="records")
    metrics_for_fips = [MetricsTimeseriesRow(**metric_row) for metric_row in metrics_timeseries]

    return metrics_for_fips, latest


def build_timeseries_for_region(
    regional_input: RegionalInput,
) -> Optional[RegionSummaryWithTimeseries]:
    fips_latest = regional_input.latest
    model_output = regional_input.model_output

    try:
        fips_timeseries = regional_input.timeseries
        metrics_timeseries, metrics_latest = generate_metrics_and_latest(
            fips_timeseries, fips_latest, model_output
        )
        region_summary = generate_api_v2.generate_region_summary(fips_latest, metrics_latest)
        region_timeseries = generate_api_v2.generate_region_timeseries(
            region_summary, fips_timeseries, metrics_timeseries
        )
    except Exception:
        logger.exception(f"Failed to build timeseries for fips.")
        return None

    return region_timeseries


def _deploy_timeseries(path_builder, timeseries):
    region_summary = timeseries.region_summary
    output_path = path_builder.single_summary(region_summary, FileType.JSON)
    deploy_json_api_output(region_summary, output_path)

    output_path = path_builder.single_timeseries(timeseries, FileType.JSON)
    deploy_json_api_output(timeseries, output_path)
    return region_summary


def deploy_single_level(all_timeseries, path_builder: APIOutputPathBuilder):
    if not all_timeseries:
        return

    all_summaries = []
    deploy_timeseries_partial = functools.partial(_deploy_timeseries, path_builder)
    all_summaries = [
        deploy_timeseries_partial(region_timeseries) for region_timeseries in all_timeseries
    ]
    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)
    bulk_summaries = AggregateRegionSummary(__root__=all_summaries)
    flattened_timeseries = generate_api_v2.generate_bulk_flattened_timeseries(bulk_timeseries)

    output_path = path_builder.bulk_timeseries(bulk_timeseries, FileType.JSON)
    deploy_json_api_output(bulk_timeseries, output_path)

    # output_path = path_builder.bulk_prediction_data(flattened_timeseries, FileType.CSV)
    # deploy_csv_api_output(flattened_timeseries, summary_folder)

    output_path = path_builder.bulk_summary(bulk_summaries, FileType.JSON)
    deploy_json_api_output(bulk_summaries, output_path)

    output_path = path_builder.bulk_summary(bulk_summaries, FileType.CSV)
    deploy_csv_api_output(bulk_summaries, output_path)


def deploy_json_api_output(
    region_result: pydantic.BaseModel, output_path: pathlib.Path,
):
    output_path.write_text(region_result.json())
    return region_result


def deploy_csv_api_output(
    api_output: pydantic.BaseModel, output_path: pathlib.Path, filename_override=None,
):
    if not hasattr(api_output, "__root__"):
        raise AssertionError("Missing root data")

    rows = dataset_deployer.remove_root_wrapper(api_output.dict())
    dataset_deployer.write_nested_csv(rows, output_path)
