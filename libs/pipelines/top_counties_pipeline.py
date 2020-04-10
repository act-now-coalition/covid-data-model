from collections import namedtuple
import logging

from libs.enums import Intervention
from libs import validate_results
from libs import build_dod_dataset
from libs import dataset_deployer

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"

TopCountiesPipelineResult = namedtuple(
    "TopCountiesPipelineResult", ["key", "projection_df"]
)


def run_projections(
    input_file, intervention: Intervention, run_validation=True
) -> TopCountiesPipelineResult:
    """Run county level projections for a specific intervention.

    Args:
        input_file: Input file to load model output results from.
        intervention: Intervention type to summarize.
        run_validation: If true runs validation on generated shapefiles
            and dataframes.

    Returns: TopCountiesPipelineResult objects for county data.
    """
    # Run County level projections
    counties_key_name = f"counties.{intervention.name}"
    counties_df = build_dod_dataset.get_usa_by_county_with_projection_df(
        input_file, intervention.value
    )
    if run_validation:
        validate_results.validate_counties_df(counties_key_name, counties_df)

    county_results = TopCountiesPipelineResult(
        counties_key_name, counties_df
    )
    return county_results


def deploy_results(intervention_result: TopCountiesPipelineResult, output: str):
    """Deploys results from an intervention to specified output directory.

    Args:
        intervention_result: Intervention result.
        output: output folder to save results in.
    """
    dataset_deployer.upload_csv(
        intervention_result.key, intervention_result.projection_df.to_csv(), output
    )