from collections import namedtuple
import logging

from libs.enums import Intervention, AggregateLevel
from libs import validate_results
from libs import build_processed_dataset
from libs import dataset_deployer
from libs.us_state_abbrev import US_STATE_ABBREV

from libs.datasets import results_schema as rc
from libs.functions import generate_api as api

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"

APIPipelineProjectionResult = namedtuple(
    "APIPipelineProjectionResult", ["intervention", "aggregate_level", "projection_df",]
)

APIPipelineAPIRow = namedtuple(
    "APIPipelineAPIRow", ["key", "api"]
)

APIPipelineResult = namedtuple("APIPipelineResult", ["api_tuples"])


def run_projections(
    input_file, aggregation_level, intervention: Intervention, run_validation=True
) -> APIPipelineProjectionResult:
    """Run the projections for the given intervention for states 
    in order to genereate the api. 

    Args:
        input_file: Input file to load model output results from.
        intervention: Intervention Enum to be used to genreate results
        run_validation: If true runs validation on generated shapefiles
            and dataframes.

    Returns: APIPipelineProjectionResult objects for county data.
    """

    if aggregation_level == AggregateLevel.STATE: 
        states_key_name = f"states.{intervention.name}"
        states_df = build_processed_dataset.get_usa_by_states_df(
            input_file, intervention.value
        )
        if run_validation:
            validate_results.validate_states_df(states_key_name, states_df)

        state_results = APIPipelineProjectionResult(intervention, AggregateLevel.STATE, states_df)
        return state_results
    elif aggregation_level == AggregateLevel.COUNTY:
        # Run County level projections
        counties_key_name = f"counties.{intervention.name}"
        counties_df = build_processed_dataset.get_usa_by_county_with_projection_df(
            input_file, intervention.value
        )
        if run_validation:
            validate_results.validate_counties_df(counties_key_name, counties_df)

        county_results = APIPipelineProjectionResult(intervention, AggregateLevel.COUNTY, counties_df)

        return county_results
    else: 
        raise Exception("Non-valid aggreation level specified")

def _get_api_prefix(aggregate_level, row): 
    if aggregate_level == AggregateLevel.COUNTY: 
        return row[rc.FIPS]
    elif aggregate_level == AggregateLevel.STATE: 
        full_state_name = row[rc.STATE]
        return US_STATE_ABBREV[full_state_name]
    else:
        raise Exception("Only County and State Aggregate Levels supported")

def generate_api(
    projection_result: APIPipelineProjectionResult, 
) -> APIPipelineResult:
    """
    pipethrough the rows of the projection
    if it's a county generate the key for counties:
        /us/counties/{FIPS_CODE}.{INTERVENTION}.json 
    if it's a state generate the key for states 
        /us/states/{STATE_ABBREV}.{INTERVENTION}.json
    """
    results = []
    for index, row in projection_result.projection_df.iterrows():
        generated_data = api.generate_api_for_projection_row(row)
        key_prefix = _get_api_prefix(projection_result.aggregate_level, row)
        generated_key = f"{key_prefix}.{projection_result.intervention.name}"
        results.append(APIPipelineAPIRow(generated_key, generated_data))
    return APIPipelineResult(results)


def deploy_results(result: APIPipelineResult, output: str):
    """Deploys results from the top counties to specified output directory.

    Args:
        result: Top Counties Pipeline result.
        key: Name for the file to be uploaded
        output: output folder to save results in.
    """
    for api_tuple in result.api_tuples: 
        dataset_deployer.upload_json(api_tuple.key, api_tuple.api.json(), output)
