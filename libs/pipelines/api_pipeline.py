from typing import Any
from typing import Dict
from typing import Iterator, List, Optional
import functools
import pathlib
from dataclasses import dataclass

import pydantic
import structlog

from api.can_api_definition import (
    AggregateRegionSummary,
    AggregateRegionSummaryWithTimeseries,
    Metrics,
    MetricsTimeseriesRow,
    RegionSummaryWithTimeseries,
)
from libs import dataset_deployer
from libs import parallel_utils
from libs import pipeline
from libs import top_level_metrics
from libs.datasets import timeseries
from libs.datasets import CommonFields
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.enums import Intervention
from libs.functions import generate_api as api
from libs.functions import get_can_projection

logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"


@dataclass(frozen=True)
class RegionalInput:
    region: pipeline.Region

    rt_data: Optional[OneRegionTimeseriesDataset]

    icu_data: Optional[OneRegionTimeseriesDataset]

    intervention: Intervention

    _combined_data: combined_datasets.RegionalData

    @property
    def fips(self) -> str:
        return self.region.fips

    @property
    def state(self) -> str:
        return self.latest[CommonFields.STATE]

    @property
    def latest(self) -> Dict[str, Any]:
        return self._combined_data.latest

    @property
    def timeseries(self) -> OneRegionTimeseriesDataset:
        return self._combined_data.timeseries

    @staticmethod
    def from_region_and_intervention(
        region: pipeline.Region,
        intervention: Intervention,
        rt_data: MultiRegionTimeseriesDataset,
        icu_data: MultiRegionTimeseriesDataset,
    ) -> "RegionalInput":
        combined_data = combined_datasets.RegionalData.from_region(region)

        try:
            rt_data = rt_data.get_one_region(region)
        except timeseries.RegionLatestNotFound:
            rt_data = None

        try:
            icu_data = icu_data.get_one_region(region)
        except timeseries.RegionLatestNotFound:
            icu_data = None

        return RegionalInput(
            region=region,
            intervention=intervention,
            _combined_data=combined_data,
            rt_data=rt_data,
            icu_data=icu_data,
        )


def run_on_all_regional_inputs_for_intervention(
    regional_inputs: List[RegionalInput], sort_func=None, limit=None,
) -> Iterator[RegionSummaryWithTimeseries]:

    # Load interventions outside of subprocesses to properly cache.
    get_can_projection.get_interventions()

    results = parallel_utils.parallel_map(build_timeseries_for_region, regional_inputs)
    all_timeseries = [region_timeseries for region_timeseries in results if region_timeseries]

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
        rt_data: Infection rate timeseries.
        icu_data: ICU timeseries.

    Returns:
        Tuple of MetricsTimeseriesRows for all days and the metrics overview.
    """
    if timeseries.empty:
        return [], None

    metrics_results, latest = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, rt_data, icu_data
    )
    metrics_timeseries = metrics_results.to_dict(orient="records")
    metrics_for_fips = [MetricsTimeseriesRow(**metric_row) for metric_row in metrics_timeseries]

    return metrics_for_fips, latest


def build_timeseries_for_region(
    regional_input: RegionalInput,
) -> Optional[RegionSummaryWithTimeseries]:
    intervention = regional_input.intervention

    if intervention is Intervention.SELECTED_INTERVENTION:
        intervention = get_can_projection.get_intervention_for_state(regional_input.state)

    try:
        metrics_timeseries, metrics_latest = generate_metrics_and_latest(
            regional_input.timeseries, regional_input.rt_data, regional_input.icu_data,
        )
        region_summary = api.generate_region_summary(regional_input.latest, metrics_latest)
        region_timeseries = api.generate_region_timeseries(
            region_summary, regional_input.timeseries, metrics_timeseries
        )
    except Exception:
        logger.exception(f"Failed to build timeseries for fips.")
        return None

    return region_timeseries


def _deploy_timeseries(intervention, region_folder, timeseries):
    region_summary = timeseries.region_summary
    deploy_json_api_output(intervention, region_summary, region_folder)
    deploy_json_api_output(intervention, timeseries, region_folder)
    return region_summary


def deploy_single_level(intervention, all_timeseries, summary_folder, region_folder):
    if not all_timeseries:
        return

    logger.info(f"Deploying {intervention.name}")

    deploy_timeseries_partial = functools.partial(_deploy_timeseries, intervention, region_folder)
    all_summaries = [
        deploy_timeseries_partial(region_timeseries) for region_timeseries in all_timeseries
    ]
    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)
    bulk_summaries = AggregateRegionSummary(__root__=all_summaries)

    deploy_json_api_output(intervention, bulk_timeseries, summary_folder)
    deploy_json_api_output(intervention, bulk_summaries, summary_folder)
    deploy_csv_api_output(intervention, bulk_summaries, summary_folder)

    flattened_timeseries = api.generate_bulk_flattened_timeseries(bulk_timeseries)
    if not flattened_timeseries.__root__:
        logger.warning(
            "No summaries, skipping deploying bulk data",
            intervention=intervention,
            summary_folder=summary_folder,
        )
        return

    deploy_csv_api_output(intervention, flattened_timeseries, summary_folder)


def deploy_json_api_output(
    intervention: Intervention,
    region_result: pydantic.BaseModel,
    output_dir: pathlib.Path,
    filename_override=None,
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    filename = filename_override or (region_result.output_key(intervention) + ".json")
    output_path = output_dir / filename
    output_path.write_text(region_result.json())
    return region_result


def deploy_csv_api_output(
    intervention: Intervention,
    api_output: pydantic.BaseModel,
    output_dir: pathlib.Path,
    filename_override=None,
):
    if not hasattr(api_output, "__root__"):
        raise AssertionError("Missing root data")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    filename = filename_override or (api_output.output_key(intervention) + ".csv")
    output_path = output_dir / filename
    rows = dataset_deployer.remove_root_wrapper(api_output.dict())
    dataset_deployer.write_nested_csv(rows, output_path)
