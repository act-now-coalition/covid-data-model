from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import multiprocessing
import pathlib
import time

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
from libs import top_level_metric_risk_levels
from libs import pipeline
from libs.datasets import timeseries
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.enums import Intervention
from libs.functions import build_api_v2
from libs.datasets import combined_datasets
from libs.datasets import AggregationLevel

logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"

INTERVENTION = Intervention.OBSERVED_INTERVENTION


@dataclass(frozen=True)
class RegionalInput:
    region: pipeline.Region

    _combined_data: combined_datasets.RegionalData

    rt_data: Optional[OneRegionTimeseriesDataset]

    icu_data: Optional[OneRegionTimeseriesDataset]

    @property
    def fips(self) -> str:
        return self.region.fips

    @property
    def latest(self) -> Dict[str, Any]:
        return self._combined_data.latest

    @property
    def timeseries(self) -> OneRegionTimeseriesDataset:
        return self._combined_data.timeseries

    @staticmethod
    def from_region_and_model_output(
        region: pipeline.Region,
        rt_data: MultiRegionTimeseriesDataset,
        icu_data: MultiRegionTimeseriesDataset,
    ) -> "RegionalInput":
        combined_data = combined_datasets.RegionalData.from_region(region)

        # Not all regions have Rt or ICU data due to various filters in pyseir code.
        try:
            rt_data = rt_data.get_one_region(region)
        except timeseries.RegionLatestNotFound:
            rt_data = None

        try:
            icu_data = icu_data.get_one_region(region)
        except timeseries.RegionLatestNotFound:
            icu_data = None

        return RegionalInput(
            region=region, _combined_data=combined_data, rt_data=rt_data, icu_data=icu_data,
        )


def run_on_regions(
    regional_inputs: List[RegionalInput],
    pool: multiprocessing.Pool = None,
    sort_func=None,
    limit=None,
) -> List[RegionSummaryWithTimeseries]:
    # Setting maxtasksperchild to one ensures that we minimize memory usage over time by creating
    # a new child for every task. Addresses OOMs we saw on highly parallel build machine.
    pool = pool or multiprocessing.Pool(maxtasksperchild=1)
    results = pool.map(build_timeseries_for_region, regional_inputs)
    all_timeseries = [result for result in results if result]

    if sort_func:
        all_timeseries.sort(key=sort_func)

    if limit:
        all_timeseries = all_timeseries[:limit]

    return all_timeseries


def generate_metrics_and_latest(
    timeseries: OneRegionTimeseriesDataset,
    rt_data: Optional[OneRegionTimeseriesDataset],
    icu_data: Optional[OneRegionTimeseriesDataset],
) -> [List[MetricsTimeseriesRow], Optional[Metrics]]:
    """
    Build metrics with timeseries.

    Args:
        timeseries: Timeseries for one region
        latest: Dictionary of latest values for region
        model_output: Optional model output for region.

    Returns:
        Tuple of MetricsTimeseriesRows for all days and the metrics overview.
    """
    if timeseries.empty:
        return [], Metrics.empty()

    metrics_results, latest = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, rt_data, icu_data
    )
    metrics_timeseries = metrics_results.to_dict(orient="records")
    metrics_for_fips = [MetricsTimeseriesRow(**metric_row) for metric_row in metrics_timeseries]

    return metrics_for_fips, latest


def build_timeseries_for_region(
    regional_input: RegionalInput,
) -> Optional[RegionSummaryWithTimeseries]:
    """Build Timeseries for a single region.

    Args:
        regional_input: Data for region.

    Returns: Summary with timeseries for region.
    """
    fips_latest = regional_input.latest

    try:
        fips_timeseries = regional_input.timeseries
        metrics_timeseries, metrics_latest = generate_metrics_and_latest(
            fips_timeseries, regional_input.rt_data, regional_input.icu_data
        )
        risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(metrics_latest)
        region_summary = build_api_v2.build_region_summary(fips_latest, metrics_latest, risk_levels)
        region_timeseries = build_api_v2.build_region_timeseries(
            region_summary, fips_timeseries, metrics_timeseries
        )
    except Exception:
        logger.exception(f"Failed to build timeseries for fips.")
        return None

    return region_timeseries


def deploy_single_level(
    all_timeseries: List[RegionSummaryWithTimeseries],
    level: AggregationLevel,
    output_root: pathlib.Path,
) -> None:
    """Deploys all files for a single aggregate level.

    Deploys individual and bulk aggregations of timeseries and summaries.

    Args:
        all_timeseries: List of timeseries to deploy.
        output_root: Root of API output.
    """

    path_builder = APIOutputPathBuilder(output_root, level)
    path_builder.make_directories()
    # Filter all timeseries to just aggregate level.
    all_timeseries = [output for output in all_timeseries if output.level is level]
    all_summaries = [output.region_summary for output in all_timeseries]

    if not all_timeseries:
        logger.warning(f"No regions detected - skipping.", aggregate_level=level.value)
        return

    logger.info(f"Deploying {level.value} output to {output_root}")

    for summary in all_summaries:
        output_path = path_builder.single_summary(summary, FileType.JSON)
        deploy_json_api_output(summary, output_path)

    for timeseries in all_timeseries:
        output_path = path_builder.single_timeseries(timeseries, FileType.JSON)
        deploy_json_api_output(timeseries, output_path)

    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)
    start = time.time()
    flattened_timeseries = build_api_v2.build_bulk_flattened_timeseries(bulk_timeseries)
    duration = time.time() - start
    logger.info(
        f"Built bulk flattened timeseries in {duration} seconds", aggregate_level=level.value
    )
    output_path = path_builder.bulk_flattened_timeseries_data(FileType.CSV)
    deploy_csv_api_output(
        flattened_timeseries, output_path, keys_to_skip=["actuals.date", "metrics.date"]
    )

    output_path = path_builder.bulk_timeseries(bulk_timeseries, FileType.JSON)
    deploy_json_api_output(bulk_timeseries, output_path)

    bulk_summaries = AggregateRegionSummary(__root__=all_summaries)
    output_path = path_builder.bulk_summary(bulk_summaries, FileType.JSON)
    deploy_json_api_output(bulk_summaries, output_path)

    output_path = path_builder.bulk_summary(bulk_summaries, FileType.CSV)
    deploy_csv_api_output(bulk_summaries, output_path)


def deploy_json_api_output(region_result: pydantic.BaseModel, output_path: pathlib.Path,) -> None:
    output_path.write_text(region_result.json())


def deploy_csv_api_output(
    api_output: pydantic.BaseModel,
    output_path: pathlib.Path,
    keys_to_skip: Optional[List[str]] = None,
) -> None:
    if not hasattr(api_output, "__root__"):
        raise AssertionError("Missing root data")

    rows = dataset_deployer.remove_root_wrapper(api_output.dict())
    dataset_deployer.write_nested_csv(rows, output_path, keys_to_skip=keys_to_skip)
