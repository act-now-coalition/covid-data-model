from typing import List, Optional, Iterator, Tuple
import pathlib
import functools
from collections import namedtuple
import logging
import multiprocessing
import pydantic
import simplejson
from api.can_api_definition import CovidActNowAreaSummary
from api.can_api_definition import CovidActNowAreaTimeseries
from api.can_api_definition import CovidActNowBulkSummary
from api.can_api_definition import CovidActNowBulkTimeseries
from api.can_api_definition import PredictionTimeseriesRowWithHeader
from libs.enums import Intervention
from libs.datasets import CommonFields
from libs.datasets.dataset_utils import AggregationLevel
from libs import validate_results
from libs import dataset_deployer
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import combined_datasets
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import results_schema as rc
from libs.functions import generate_api as api
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


def run_on_all_fips_for_intervention(
    latest_values: LatestValuesDataset,
    timeseries: TimeseriesDataset,
    intervention: Intervention,
    model_output_dir: pathlib.Path,
    pool: multiprocessing.Pool = None,
) -> Iterator[CovidActNowAreaTimeseries]:
    # def run_fips(fips):
    #     return build_summary_and_timeseries_for_fips(
    #         fips, intervention, latest_values, timeseries, model_output_dir
    #     )

    run_fips = functools.partial(
        build_summary_and_timeseries_for_fips,
        intervention,
        latest_values,
        timeseries,
        model_output_dir,
    )

    pool = pool or multiprocessing.Pool()
    results = pool.map(run_fips, latest_values.all_fips)

    for area_summary, area_timeseries in results:
        if area_timeseries:
            yield area_summary, area_timeseries


def build_summary_and_timeseries_for_fips(
    intervention, us_latest, us_timeseries, model_output_dir, fips,
) -> Tuple[Optional[CovidActNowAreaSummary], Optional[CovidActNowAreaTimeseries]]:
    model_output = CANPyseirLocationOutput.load_from_model_output_if_exists(
        fips, intervention, model_output_dir
    )
    if not model_output and intervention is not Intervention.OBSERVED_INTERVENTION:
        # All model output is currently tied to a specific intervention. However,
        # we want to generate results for areas that don't have a fit result, but we're not
        # duplicating non-model outputs.
        return None, None

    fips_latest = us_latest.get_record_for_fips(fips)

    try:
        area_summary = api.generate_area_summary(intervention, fips_latest, model_output)
        fips_timeseries = us_timeseries.get_subset(None, fips=fips)
        area_timeseries = api.generate_area_timeseries(area_summary, fips_timeseries, model_output)
    except Exception:
        logger.exception("failed to run output")
        return None, None

    return area_summary, area_timeseries


def remove_root_wrapper(obj: dict):
    """Removes __root__ and replaces with __root__ value.

    When pydantic models are used to wrap lists this is done using a property __root__.
    When this is serialized using `model.json()`, it will return a json list. However,
    calling `model.dict()` will return a dictionary with a single key `__root__`.
    This function removes that __root__ key (and all sub pydantic models with a
    similar structure) to have a similar hierarchy to the json output.

    A dictionary {"__root__": []} will return [].

    Args:
        obj: pydantic model as dict.

    Returns: object with __root__ removed.
    """
    # Objects with __root__ should have it as the only key.
    if len(obj) == 1 and "__root__" in obj:
        return obj["__root__"]

    results = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            value = remove_root_wrapper(value)

        results[key] = value

    return results


def deploy_single_region(
    intervention: Intervention, area_result: CovidActNowAreaSummary, output_dir: pathlib.Path
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (area_result.output_key(intervention) + ".json")
    output_path.write_text(area_result.json())
    return area_result


# def deploy_results(results: List[APIOutput], output: str, write_csv=False):
#     """Deploys results from the top counties to specified output directory.

#     Args:
#         result: Top Counties Pipeline result.
#         key: Name for the file to be uploaded
#         output: output folder to save results in.
#     """
#     output_path = pathlib.Path(output)
#     if not output_path.exists():
#         output_path.mkdir(parents=True, exist_ok=True)

#     for api_row in results:
#         data = remove_root_wrapper(api_row.data.dict())
#         # Encoding approach based on Pydantic's implementation of .json():
#         # https://github.com/samuelcolvin/pydantic/pull/210/files
#         # `json` isn't in `pydantic/__init__py` which I think means it doesn't intend to export
#         # it. We use it anyway and pylint started complaining.
#         # pylint: disable=no-member
#         data_as_json = simplejson.dumps(
#             data, ignore_nan=True, default=pydantic.json.pydantic_encoder
#         )
#         dataset_deployer.upload_json(api_row.file_stem, data_as_json, output)
#         if write_csv:
#             if not isinstance(data, list):
#                 raise ValueError("Cannot find list data for csv export.")
#             dataset_deployer.write_nested_csv(data, api_row.file_stem, output)


# def build_prediction_header_timeseries_data(data: APIOutput):

#     rows = []
#     api_data = data.data
#     # Iterate through each state or county in data, adding summary data to each
#     # timeseries row.
#     for row in api_data.__root__:
#         county_name = None
#         if isinstance(row, CovidActNowCountyTimeseries):
#             county_name = row.countyName

#         summary_data = {
#             "countryName": row.countryName,
#             "countyName": county_name,
#             "stateName": row.stateName,
#             "fips": row.fips,
#             "lat": row.lat,
#             "long": row.long,
#             "intervention": data.intervention.name,
#             "lastUpdatedDate": row.lastUpdatedDate,
#         }

#         for timeseries_data in row.timeseries:
#             timeseries_row = PredictionTimeseriesRowWithHeader(
#                 **summary_data, **timeseries_data.dict()
#             )
#             rows.append(timeseries_row)

#     return APIOutput(data.file_stem, rows, data.intervention)


# def deploy_prediction_timeseries_csvs(data: APIOutput, output):
#     dataset_deployer.write_nested_csv([row.dict() for row in data.data], data.file_stem, output)
