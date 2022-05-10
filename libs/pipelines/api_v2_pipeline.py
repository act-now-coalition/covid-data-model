from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pathlib
import pandas as pd
import pydantic
import structlog

import pyseir.run
from libs.metrics import community_levels, test_positivity
from libs.pipelines.api_v2_paths import APIOutputPathBuilder
from libs.pipelines.api_v2_paths import FileType
from libs.pipelines import csv_column_ordering
from api.can_api_v2_definition import AggregateRegionSummary
from api.can_api_v2_definition import AggregateRegionSummaryWithTimeseries
from api.can_api_v2_definition import Metrics
from api.can_api_v2_definition import RegionSummaryWithTimeseries
from api.can_api_v2_definition import RegionSummary
from libs import dataset_deployer
from libs.metrics import top_level_metrics
from libs.metrics import top_level_metric_risk_levels
from libs.metrics import cdc_transmission_levels

from libs import parallel_utils
from libs import pipeline
from libs import build_api_v2
from libs.datasets import timeseries
from libs.datasets import vaccine_backfills
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets import AggregationLevel
from libs.timing_utils import timing


logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"


@dataclass(frozen=True)
class RegionalInput:
    region: pipeline.Region

    _combined_data_with_test_positivity: OneRegionTimeseriesDataset

    rt_data: Optional[OneRegionTimeseriesDataset]

    @property
    def fips(self) -> str:
        return self.region.fips

    @property
    def latest(self) -> Dict[str, Any]:
        return self._combined_data_with_test_positivity.latest

    @property
    def timeseries(self) -> OneRegionTimeseriesDataset:
        return self._combined_data_with_test_positivity

    @staticmethod
    def from_region_and_model_output(
        region: pipeline.Region,
        combined_data_with_test_positivity: MultiRegionDataset,
        rt_data: MultiRegionDataset,
    ) -> "RegionalInput":
        one_region_data = combined_data_with_test_positivity.get_one_region(region)

        # Not all regions have Rt or ICU data due to various filters in pyseir code.
        try:
            rt_data = rt_data.get_one_region(region)
        except timeseries.RegionLatestNotFound:
            rt_data = None

        return RegionalInput(
            region=region, _combined_data_with_test_positivity=one_region_data, rt_data=rt_data,
        )

    @staticmethod
    def from_one_regions(
        region: pipeline.Region,
        regional_data: OneRegionTimeseriesDataset,
        rt_data: Optional[OneRegionTimeseriesDataset],
    ):
        return RegionalInput(
            region=region, _combined_data_with_test_positivity=regional_data, rt_data=rt_data,
        )


def run_on_regions(
    regional_inputs: List[RegionalInput], sort_func=None, limit=None,
) -> List[RegionSummaryWithTimeseries]:
    results = parallel_utils.parallel_map(build_timeseries_for_region, regional_inputs)
    all_timeseries = [result for result in results if result]

    if sort_func:
        all_timeseries.sort(key=sort_func)

    if limit:
        all_timeseries = all_timeseries[:limit]

    return all_timeseries


def generate_metrics_and_latest(
    timeseries: OneRegionTimeseriesDataset, rt_data: Optional[OneRegionTimeseriesDataset], log,
) -> [pd.DataFrame, Metrics]:
    """
    Build metrics with timeseries.

    Args:
        timeseries: Timeseries for one region.
        rt_data: Rt data.


    Returns:
        Tuple of MetricsTimeseriesRows for all days and the metrics overview.
    """
    if timeseries.empty:
        return pd.DataFrame([]), Metrics.empty()

    metrics_results, latest = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, rt_data, log
    )
    return metrics_results, latest


def build_timeseries_for_region(
    regional_input: RegionalInput,
) -> Optional[RegionSummaryWithTimeseries]:
    """Build Timeseries for a single region.

    Args:
        regional_input: Data for region.

    Returns: Summary with timeseries for region.
    """
    log = structlog.get_logger(location_id=regional_input.region.location_id)

    try:
        fips_timeseries = regional_input.timeseries
        metrics_results, metrics_latest = generate_metrics_and_latest(
            fips_timeseries, regional_input.rt_data, log
        )
        risk_timeseries = top_level_metric_risk_levels.calculate_risk_level_timeseries(
            metrics_results
        )
        risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(metrics_latest)

        cdc_transmission_level_timeseries = cdc_transmission_levels.calculate_transmission_level_timeseries(
            metrics_results
        )
        cdc_transmission_level = cdc_transmission_levels.calculate_transmission_level_from_metrics(
            metrics_latest
        )

        (
            community_levels_timeseries,
            latest_community_levels,
        ) = community_levels.calculate_community_level_timeseries_and_latest(
            fips_timeseries, metrics_results
        )

        region_summary = build_api_v2.build_region_summary(
            regional_input.timeseries,
            metrics_latest,
            risk_levels,
            cdc_transmission_level,
            latest_community_levels,
            log,
        )
        region_timeseries = build_api_v2.build_region_timeseries(
            region_summary,
            fips_timeseries,
            metrics_results,
            risk_timeseries,
            cdc_transmission_level_timeseries,
            community_levels_timeseries,
        )
    except Exception:
        log.exception(f"Failed to build timeseries for fips.")
        return None

    return region_timeseries


@timing
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
        # TODO(chris): Remove this state specific block and produce timeseries csvs for all
        # individual regions.
        # Currently this is slow and adds a decent amount of time to the pipeline (~10 minutes
        # I think) if ran on all regions.
        if level in [AggregationLevel.STATE, AggregationLevel.COUNTRY]:
            output_path = path_builder.single_timeseries(timeseries, FileType.CSV)
            bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=[timeseries])
            flattened_timeseries = build_api_v2.build_bulk_flattened_timeseries(bulk_timeseries)
            deploy_csv_api_output(
                flattened_timeseries, output_path, csv_column_ordering.TIMESERIES_ORDER
            )

    deploy_bulk_files(path_builder, all_timeseries, all_summaries)

    if level is AggregationLevel.COUNTY:
        for state in set(record.state for record in all_summaries):
            state_timeseries = [record for record in all_timeseries if record.state == state]
            state_summaries = [record for record in all_summaries if record.state == state]
            deploy_bulk_files(path_builder, state_timeseries, state_summaries, state=state)


def deploy_bulk_files(
    path_builder,
    all_timeseries: List[RegionSummaryWithTimeseries],
    all_summaries: List[RegionSummary],
    state: Optional[str] = None,
):

    timing_kwargs = {"region_level": path_builder.level.value, "state": state}

    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)
    bulk_summaries = AggregateRegionSummary(__root__=all_summaries)

    flattened_timeseries = build_api_v2.build_bulk_flattened_timeseries(bulk_timeseries)

    output_path = path_builder.bulk_flattened_timeseries_data(FileType.CSV, state=state)
    deploy_csv_api_output(flattened_timeseries, output_path, csv_column_ordering.TIMESERIES_ORDER)

    output_path = path_builder.bulk_timeseries(bulk_timeseries, FileType.JSON, state=state)
    deploy_json_api_output(bulk_timeseries, output_path)

    output_path = path_builder.bulk_summary(bulk_summaries, FileType.JSON, state=state)
    deploy_json_api_output(bulk_summaries, output_path)

    output_path = path_builder.bulk_summary(bulk_summaries, FileType.CSV, state=state)

    # States were the only summary file to contain all icu headroom details
    summary_order = csv_column_ordering.SUMMARY_ORDER_NO_HEADROOM_DETAILS
    if path_builder.level is AggregationLevel.STATE:
        summary_order = csv_column_ordering.SUMMARY_ORDER

    deploy_csv_api_output(bulk_summaries, output_path, summary_order)


def deploy_json_api_output(region_result: pydantic.BaseModel, output_path: pathlib.Path) -> None:
    # Excluding fields that are not specifically included in a model.
    # This lets a field be undefined and not included in the actual json.
    serialized_result = region_result.json(exclude_unset=True)

    output_path.write_text(serialized_result)


def _model_to_dict(data: dict):
    results = {}
    for key, value in data.items():
        if isinstance(value, pydantic.BaseModel):
            value = _model_to_dict(value.__dict__)

        if isinstance(value, list):
            values = []
            for item in value:
                if isinstance(item, pydantic.BaseModel):
                    value = _model_to_dict(item.__dict__)

                values.append(value)
            value = values

        results[key] = value

    return results


def deploy_csv_api_output(
    api_output: pydantic.BaseModel, output_path: pathlib.Path, columns: List[str]
) -> None:
    if not hasattr(api_output, "__root__"):
        raise AssertionError("Missing root data")

    data = _model_to_dict(api_output.__dict__)
    rows = dataset_deployer.remove_root_wrapper(data)
    dataset_deployer.write_nested_csv(rows, output_path, header=columns)


def generate_from_loaded_data(
    model_output: pyseir.run.PyseirOutputDatasets,
    output: pathlib.Path,
    selected_dataset: MultiRegionDataset,
    log,
):
    """Runs the API generation code using data in parameters, writing results to output."""
    # If calculating test positivity succeeds join it with the combined_datasets into one
    # MultiRegionDataset
    log.info("Running test positivity.")
    regions_data = test_positivity.run_and_maybe_join_columns(selected_dataset, log)

    regions_data = vaccine_backfills.derive_vaccine_pct(regions_data)

    log.info(f"Joining inputs by region.")
    rt_data_map = dict(model_output.infection_rate.iter_one_regions())
    regional_inputs = [
        RegionalInput.from_one_regions(region, regional_data, rt_data=rt_data_map.get(region),)
        for region, regional_data in regions_data.iter_one_regions()
    ]
    # Build all region timeseries API Output objects.
    log.info("Generating all API Timeseries")
    all_timeseries = run_on_regions(regional_inputs)
    deploy_single_level(all_timeseries, AggregationLevel.COUNTY, output)
    deploy_single_level(all_timeseries, AggregationLevel.STATE, output)
    deploy_single_level(all_timeseries, AggregationLevel.CBSA, output)
    deploy_single_level(all_timeseries, AggregationLevel.PLACE, output)
    deploy_single_level(all_timeseries, AggregationLevel.COUNTRY, output)
    log.info("Finished API generation.")
