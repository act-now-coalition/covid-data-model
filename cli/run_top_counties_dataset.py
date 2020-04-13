#!/usr/bin/env python
import click
import logging

from libs.pipelines import top_counties_pipeline

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-top-counties")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of state/county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/top_counties",
    help="Output directory for artifacts",
)
def deploy_top_counties(disable_validation, input_dir, output):
    """The entry function for invocation"""

    county_result = top_counties_pipeline.run_projections(
        input_dir, run_validation=not disable_validation
    )
    county_results_api = top_counties_pipeline.generate_api(county_result)
    top_counties_pipeline.deploy_results(county_results_api, "counties_top_100", output)

    logger.info("finished top counties job")
