from typing import Iterator, List, Optional
import functools
import multiprocessing
import pathlib

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
from libs import top_level_metrics
from libs.datasets import CommonFields
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention
from libs.functions import generate_api as api
from libs.functions import get_can_projection

logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"


def run_on_all_fips_for_intervention(
    latest_values: LatestValuesDataset,
    timeseries: TimeseriesDataset,
    intervention: Intervention,
    model_output_dir: pathlib.Path,
    pool: multiprocessing.Pool = None,
    sort_func=None,
    limit=None,
) -> Iterator[RegionSummaryWithTimeseries]:
    run_fips = functools.partial(
        build_timeseries_for_fips, intervention, latest_values, timeseries, model_output_dir
    )

    # Load interventions outside of subprocesses to properly cache.
    get_can_projection.get_interventions()

    # Setting maxtasksperchild to one ensures that we minimize memory usage over time by creating
    # a new child for every task. Addresses OOMs we saw on highly parallel build machine.
    pool = pool or multiprocessing.Pool(maxtasksperchild=1)
    results = pool.map(run_fips, latest_values.all_fips)
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


def generate_metrics_and_latest_for_fips(
    timeseries: TimeseriesDataset, latest: dict, model_output: Optional[CANPyseirLocationOutput],
) -> [List[MetricsTimeseriesRow], Optional[Metrics]]:
    """
    For a FIPS, generate a MetricsTimeseriesRow per day and return the latest.

    Args:
        fips: FIPS to run on.

    Returns:
        Tuple of MetricsTimeseriesRows for all days and the latest.
    """
    metrics_results = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, latest, model_output
    )
    metrics_timeseries = metrics_results.to_dict(orient="records")
    metrics_for_fips = [MetricsTimeseriesRow(**metric_row) for metric_row in metrics_timeseries]
    latest = None
    if metrics_for_fips:
        latest = top_level_metrics.calculate_latest_metrics(metrics_results)

    return metrics_for_fips, latest


def build_timeseries_for_fips(
    intervention, us_latest, us_timeseries, model_output_dir, fips
) -> Optional[RegionSummaryWithTimeseries]:
    fips_latest = us_latest.get_record_for_fips(fips)

    if intervention is Intervention.SELECTED_INTERVENTION:
        state = fips_latest[CommonFields.STATE]
        intervention = get_can_projection.get_intervention_for_state(state)

    model_output = CANPyseirLocationOutput.load_from_model_output_if_exists(
        fips, intervention, model_output_dir
    )
    if not model_output and intervention is not Intervention.OBSERVED_INTERVENTION:
        # All model output is currently tied to a specific intervention. However,
        # we want to generate results for regions that don't have a fit result, but we're not
        # duplicating non-model outputs.
        return None

    try:
        fips_timeseries = us_timeseries.get_subset(None, fips=fips)
        metrics_timeseries, metrics_latest = generate_metrics_and_latest_for_fips(
            fips_timeseries, fips_latest, model_output
        )
        region_summary = api.generate_region_summary(fips_latest, metrics_latest, model_output)
        region_timeseries = api.generate_region_timeseries(
            region_summary, fips_timeseries, metrics_timeseries, model_output
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

    all_summaries = []
    deploy_timeseries_partial = functools.partial(_deploy_timeseries, intervention, region_folder)
    all_summaries = [
        deploy_timeseries_partial(region_timeseries) for region_timeseries in all_timeseries
    ]
    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)
    bulk_summaries = AggregateRegionSummary(__root__=all_summaries)
    flattened_timeseries = api.generate_bulk_flattened_timeseries(bulk_timeseries)

    deploy_json_api_output(intervention, bulk_timeseries, summary_folder)
    deploy_csv_api_output(intervention, flattened_timeseries, summary_folder)

    deploy_json_api_output(intervention, bulk_summaries, summary_folder)
    deploy_csv_api_output(intervention, bulk_summaries, summary_folder)


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
