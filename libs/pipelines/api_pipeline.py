from collections import namedtuple
import logging

from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel
from libs import validate_results
from libs import build_processed_dataset
from libs import dataset_deployer
from libs.us_state_abbrev import US_STATE_ABBREV

from libs.datasets import results_schema as rc
from libs.functions import generate_api as api

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"

APIPipelineProjectionResult = namedtuple(
    "APIPipelineProjectionResult",
    ["intervention", "aggregation_level", "projection_df",],
)

APIGenerationRow = namedtuple("APIGenerationRow", ["key", "api"])

APIGeneration = namedtuple("APIGeneration", ["api_rows"])


def _get_api_prefix(aggregation_level, row):
    if aggregation_level == AggregationLevel.COUNTY:
        return row[rc.FIPS]
    elif aggregation_level == AggregationLevel.STATE:
        full_state_name = row[rc.STATE]
        return US_STATE_ABBREV[full_state_name]
    else:
        raise ValueError("Only County and State Aggregate Levels supported")


def run_projections(
    input_file, aggregation_level, intervention: Intervention, run_validation=True
) -> APIPipelineProjectionResult:
    """Run the projections for the given intervention for states 
    in order to generate the api. 

    Args:
        input_file: Input file to load model output results from.
        intervention: Intervention Enum to be used to generate results
        run_validation: If true runs validation on generated shapefiles
            and dataframes.

    Returns: APIPipelineProjectionResult objects for county data.
    """

    if aggregation_level == AggregationLevel.STATE:
        states_key_name = f"states.{intervention.name}"
        states_df = build_processed_dataset.get_usa_by_states_df(
            input_file, intervention.value
        )
        if run_validation:
            validate_results.validate_states_df(states_key_name, states_df)

        state_results = APIPipelineProjectionResult(
            intervention, AggregationLevel.STATE, states_df
        )
        return state_results
    elif aggregation_level == AggregationLevel.COUNTY:
        # Run County level projections
        counties_key_name = f"counties.{intervention.name}"
        counties_df = build_processed_dataset.get_usa_by_county_with_projection_df(
            input_file, intervention.value
        )
        if run_validation:
            validate_results.validate_counties_df(counties_key_name, counties_df)

        county_results = APIPipelineProjectionResult(
            intervention, AggregationLevel.COUNTY, counties_df
        )
        return county_results
    else:
        raise ValueError("Non-valid aggreation level specified")


def _generate_api_without_ts(projection_result, row, input_dir):
    if projection_result.aggregation_level == AggregationLevel.STATE:
        generated_data = api.generate_api_for_state_projection_row(
            row
        )
    elif projection_result.aggregation_level == AggregationLevel.COUNTY:
        generated_data = api.generate_api_for_county_projection_row(
            row
        )
    else:
        raise ValueError("Aggregate Level not supported by api generation")
    key_prefix = _get_api_prefix(projection_result.aggregation_level, row)
    generated_key = f"{key_prefix}.{projection_result.intervention.name}"
    return APIGenerationRow(generated_key, generated_data)


def _generate_api_with_ts(projection_result, row, input_dir):
    if projection_result.aggregation_level == AggregationLevel.STATE:
        generated_data = api.generate_api_for_state_timeseries(
            row, projection_result.intervention, input_dir
        )
    elif projection_result.aggregation_level == AggregationLevel.COUNTY:
        generated_data = api.generate_api_for_county_timeseries(
            row, projection_result.intervention, input_dir
        )
    else:
        raise ValueError("Aggregate Level not supported by api generation")
    key_prefix = _get_api_prefix(projection_result.aggregation_level, row)
    generated_key = f"{key_prefix}.{projection_result.intervention.name}.timeseries"
    return APIGenerationRow(generated_key, generated_data)


def generate_api(
    projection_result: APIPipelineProjectionResult, input_dir: str
) -> APIGeneration:
    """
    pipethrough the rows of the projection
    if it's a county generate the key for counties:
        /us/counties/{FIPS_CODE}.{INTERVENTION}.json 
    if it's a state generate the key for states 
        /us/states/{STATE_ABBREV}.{INTERVENTION}.json
    """
    results = []
    for index, row in projection_result.projection_df.iterrows():
        results.append(_generate_api_without_ts(projection_result, row, input_dir))
        results.append(_generate_api_with_ts(projection_result, row, input_dir))
    return APIGeneration(results)


def deploy_results(result: APIGeneration, output: str):
    """Deploys results from the top counties to specified output directory.

    Args:
        result: Top Counties Pipeline result.
        key: Name for the file to be uploaded
        output: output folder to save results in.
    """
    for api_row in result.api_rows:
        dataset_deployer.upload_json(api_row.key, api_row.api.json(), output)
