from typing import List, Optional, Iterator, Tuple
import pathlib
import functools
from collections import namedtuple
import logging
import structlog
import multiprocessing
import pydantic
import simplejson
from libs.functions import get_can_projection
from api.can_api_definition import CovidActNowAreaSummary
from api.can_api_definition import CovidActNowAreaTimeseries
from api.can_api_definition import CovidActNowBulkSummary
from api.can_api_definition import CovidActNowBulkTimeseries
from api.can_api_definition import CovidActNowBulkFlattenedTimeseries
from api.can_api_definition import PredictionTimeseriesRowWithHeader
from libs.enums import Intervention
from libs.datasets import CommonFields
from libs.datasets.dataset_utils import AggregationLevel
from libs import dataset_deployer
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import combined_datasets
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import results_schema as rc
from libs.functions import generate_api as api
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput

logger = structlog.getLogger()
PROD_BUCKET = "data.covidactnow.org"


def run_on_all_fips_for_intervention(
    latest_values: LatestValuesDataset,
    timeseries: TimeseriesDataset,
    intervention: Intervention,
    model_output_dir: pathlib.Path,
    pool: multiprocessing.Pool = None,
) -> Iterator[CovidActNowAreaTimeseries]:
    run_fips = functools.partial(
        build_timeseries_for_fips, intervention, latest_values, timeseries, model_output_dir,
    )

    # Load interventions outside of subprocesses to properly cache.
    get_can_projection.get_interventions()
    pool = pool or multiprocessing.Pool()
    results = pool.map(run_fips, latest_values.all_fips)
    all_timeseries = []

    for area_timeseries in results:
        if not area_timeseries:
            continue

        all_timeseries.append(area_timeseries)

    return all_timeseries


def build_timeseries_for_fips(
    intervention, us_latest, us_timeseries, model_output_dir, fips,
) -> Optional[CovidActNowAreaTimeseries]:
    fips_latest = us_latest.get_record_for_fips(fips)

    if intervention is Intervention.SELECTED_INTERVENTION:
        state = fips_latest[CommonFields.STATE]
        intervention = get_can_projection.get_intervention_for_state(state)

    model_output = CANPyseirLocationOutput.load_from_model_output_if_exists(
        fips, intervention, model_output_dir
    )
    if not model_output and intervention is not Intervention.OBSERVED_INTERVENTION:
        # All model output is currently tied to a specific intervention. However,
        # we want to generate results for areas that don't have a fit result, but we're not
        # duplicating non-model outputs.
        return None

    try:
        area_summary = api.generate_area_summary(fips_latest, model_output)
        fips_timeseries = us_timeseries.get_subset(None, fips=fips)
        area_timeseries = api.generate_area_timeseries(area_summary, fips_timeseries, model_output)
    except Exception:
        logger.error(f"failed to run output", fips=fips)
        raise
        return None

    return area_timeseries


def deploy_single_level(intervention, all_timeseries, summary_folder, region_folder):
    if not all_timeseries:
        return
    all_summaries = []
    for timeseries in all_timeseries:
        area_summary = timeseries.area_summary
        all_summaries.append(area_summary)
        deploy_json_api_output(intervention, area_summary, region_folder)
        deploy_json_api_output(intervention, timeseries, region_folder)

    bulk_timeseries = CovidActNowBulkTimeseries(__root__=all_timeseries)
    bulk_summaries = CovidActNowBulkSummary(__root__=all_summaries)
    flattened_timeseries = api.generate_bulk_flattened_timeseries(bulk_timeseries)

    deploy_json_api_output(intervention, bulk_timeseries, summary_folder)
    deploy_json_api_output(intervention, bulk_summaries, summary_folder)
    deploy_csv_api_output(intervention, bulk_summaries, summary_folder)
    deploy_csv_api_output(intervention, flattened_timeseries, summary_folder)


def deploy_json_api_output(
    intervention: Intervention, area_result: pydantic.BaseModel, output_dir: pathlib.Path
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (area_result.output_key(intervention) + ".json")
    output_path.write_text(area_result.json())
    return area_result


def deploy_csv_api_output(
    intervention: Intervention, api_output: pydantic.BaseModel, output_dir: pathlib.Path
):
    if not hasattr(api_output, "__root__"):
        raise AssertionError("Missing root data")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (api_output.output_key(intervention) + ".csv")
    rows = dataset_deployer.remove_root_wrapper(api_output.dict())
    dataset_deployer.write_nested_csv(rows, output_path)


def deploy_bulk_summary(
    intervention: Intervention, data: pydantic.BaseModel, output_dir: pathlib.Path, as_csv=False
):

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (data.output_key(intervention) + ".json")
    json_rows = dataset_deployer.remove_root_wrapper(data.json())
    output_path.write_text(area_result.json())
