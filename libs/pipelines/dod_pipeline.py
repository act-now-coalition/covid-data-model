from typing import Tuple
from collections import namedtuple
import logging

from libs.enums import Intervention
from libs import validate_results
from libs import build_processed_dataset
from libs import dataset_deployer
from libs.functions import generate_shapefiles

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"

DodInterventionResult = namedtuple(
    "DodInterventionResult", ["key", "projection_df", "shapefile_data"]
)


def run_projections(
    state_input_file, county_input_file, intervention: Intervention, run_validation=True
) -> Tuple[DodInterventionResult, DodInterventionResult]:
    """Run county and state level projections for a specific intervention.

    Args:
        input_file: Input file to load model output results from.
        intervention: Intervention type to summarize.
        run_validation: If true runs validation on generated shapefiles
            and dataframes.

    Returns: Tuple of DodInterventionResult objects for state and county data.
    """
    states_key_name = f"states.{intervention.name}"
    states_df = build_processed_dataset.get_usa_by_states_df(
        state_input_file, intervention.value
    )
    if run_validation:
        validate_results.validate_states_df(states_key_name, states_df)

    states_shp, states_shx, states_dbf = generate_shapefiles.get_usa_state_shapefile(
        states_df
    )
    if run_validation:
        validate_results.validate_states_shapefile(
            states_key_name, states_shp, states_shx, states_dbf
        )
    logger.info(f"Generated state shape files for {intervention.name}")

    # Run County level projections
    counties_key_name = f"counties.{intervention.name}"
    counties_df = build_processed_dataset.get_usa_by_county_with_projection_df(
        county_input_file, intervention.value
    )
    if run_validation:
        validate_results.validate_counties_df(counties_key_name, counties_df)

    (
        counties_shp,
        counties_shx,
        counties_dbf,
    ) = generate_shapefiles.get_usa_county_shapefile(counties_df)
    if run_validation:
        validate_results.validate_counties_shapefile(
            counties_key_name, counties_shp, counties_shx, counties_dbf
        )

    state_results = DodInterventionResult(
        states_key_name, states_df, (states_shp, states_shx, states_dbf)
    )

    county_results = DodInterventionResult(
        counties_key_name, counties_df, (counties_shp, counties_shx, counties_dbf)
    )
    return state_results, county_results


def deploy_results(intervention_result: DodInterventionResult, output: str):
    """Deploys results from an intervention to specified output directory.

    Args:
        intervention_result: Intervention result.
        output: output folder to save results in.
    """
    dataset_deployer.upload_csv(
        intervention_result.key, intervention_result.projection_df.to_csv(), output
    )
    dataset_deployer.deploy_shape_files(
        output, intervention_result.key, *intervention_result.shapefile_data
    )
    logger.info(f"Generated state shape files")
