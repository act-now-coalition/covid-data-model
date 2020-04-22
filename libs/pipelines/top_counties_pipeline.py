from collections import namedtuple
import logging

from libs.enums import Intervention
from libs import validate_results
from libs import build_processed_dataset
from libs import dataset_deployer

from libs.datasets import results_schema as rc
from libs.functions import generate_api as api

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"

TopCountiesPipelineProjectionResult = namedtuple(
    "TopCountiesPipelineProjectionResult", ["projection_df",]
)

TopCountiesPipelineResult = namedtuple("TopCountiesPipelineResult", ["api"])


def run_projections(
    input_file, run_validation=True
) -> TopCountiesPipelineProjectionResult:
    """Run the projections for the current intervention for counties
    in order to genereate a list of the 100 counties most affected

    Args:
        input_file: Input file to load model output results from.
        run_validation: If true runs validation on generated shapefiles
            and dataframes.

    Returns: TopCountiesPipelineProjectionResult objects for county data.
    """
    # Run County level projections
    intervention = Intervention.CURRENT_INTERVENTION

    counties_key_name = f"counties.{intervention.name}"
    # note i think build_processed_dataset should porbably be renamed?
    counties_df = build_processed_dataset.get_usa_by_county_with_projection_df(
        input_file, intervention.value
    )
    if run_validation:
        validate_results.validate_counties_df(counties_key_name, counties_df)

    county_results = TopCountiesPipelineProjectionResult(counties_df)

    return county_results


def generate_api(
    projection_result: TopCountiesPipelineProjectionResult,
    sort_fields=[rc.PEAK_HOSPITALIZATION_SHORTFALL],
    length=100,
) -> TopCountiesPipelineResult:
    projection = projection_result.projection_df
    sorted_limited = projection.sort_values(by=sort_fields, ascending=False).head(
        length
    )
    return TopCountiesPipelineResult(
        api.generate_api_for_county_projection(sorted_limited)
    )


def deploy_results(result: TopCountiesPipelineResult, key: str, output: str):
    """Deploys results from the top counties to specified output directory.

    Args:
        result: Top Counties Pipeline result.
        key: Name for the file to be uploaded
        output: output folder to save results in.
    """
    dataset_deployer.upload_json(key, result.api.json(), output)
